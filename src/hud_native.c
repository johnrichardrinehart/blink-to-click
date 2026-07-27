#define _POSIX_C_SOURCE 200809L

#include "wlr-layer-shell-unstable-v1-client-protocol.h"
#include "xdg-output-unstable-v1-client-protocol.h"

#include <cairo.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pango/pangocairo.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <wayland-client.h>

#define HUD_WIDTH 540
#define HUD_HEIGHT 112
#define HUD_BUFFER_COUNT 3
#define HUD_MARGIN 16
#define HUD_RADIUS 9.0
#define HUD_MAX_OUTPUTS 32
#define TRAINING_NAMESPACE "gazeebo-training"

typedef struct HudState HudState;

typedef struct {
  struct wl_buffer *buffer;
  uint32_t *pixels;
  bool busy;
} HudBuffer;

typedef struct {
  struct wl_output *output;
  struct zxdg_output_v1 *xdg_output;
  uint32_t registry_name;
  bool active;
  int32_t x;
  int32_t y;
  int32_t width;
  int32_t height;
  int32_t physical_width_mm;
  int32_t physical_height_mm;
  int32_t current_mode_width;
  int32_t current_mode_height;
  int32_t transform;
  bool has_current_mode;
  char name[128];
  char description[384];
} HudOutput;

struct HudState {
  struct wl_display *display;
  struct wl_registry *registry;
  struct wl_compositor *compositor;
  struct wl_shm *shm;
  struct zwlr_layer_shell_v1 *layer_shell;
  struct zxdg_output_manager_v1 *output_manager;
  struct wl_shm_pool *pool;
  struct wl_surface *surface;
  struct zwlr_layer_surface_v1 *layer_surface;
  void *pool_data;
  size_t pool_size;
  int32_t frame_width;
  int32_t frame_height;
  HudBuffer buffers[HUD_BUFFER_COUNT];
  HudOutput outputs[HUD_MAX_OUTPUTS];
  size_t output_count;
  HudOutput *training_output;
  bool training_metrics_exact;
  int next_buffer;
  bool configured;
  bool closed;
  bool failed;
  char error[256];
};

static void set_error(HudState *state, const char *message) {
  if (!state->failed) {
    snprintf(state->error, sizeof(state->error), "%s", message);
  }
  state->failed = true;
}

static void copy_error(const HudState *state, char *error, size_t error_size) {
  if (error != NULL && error_size > 0) {
    snprintf(error, error_size, "%s", state->error);
  }
}

static int anonymous_file(size_t size) {
  const char *runtime = getenv("XDG_RUNTIME_DIR");
  if (runtime == NULL || runtime[0] == '\0') {
    errno = ENOENT;
    return -1;
  }
  char path[4096];
  snprintf(path, sizeof(path), "%s/gazeebo-renderer-XXXXXX", runtime);
  int fd = mkstemp(path);
  if (fd < 0) {
    return -1;
  }
  unlink(path);
  int flags = fcntl(fd, F_GETFD);
  if (flags >= 0) {
    (void)fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
  }
  if (ftruncate(fd, (off_t)size) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

static void buffer_release(void *data, struct wl_buffer *buffer) {
  (void)buffer;
  ((HudBuffer *)data)->busy = false;
}

static const struct wl_buffer_listener BUFFER_LISTENER = {
    .release = buffer_release,
};

static void wl_output_geometry(void *data, struct wl_output *wl_output,
                               int32_t x, int32_t y, int32_t physical_width,
                               int32_t physical_height, int32_t subpixel,
                               const char *make, const char *model,
                               int32_t transform) {
  (void)wl_output;
  (void)x;
  (void)y;
  (void)subpixel;
  (void)make;
  (void)model;
  HudOutput *output = data;
  output->physical_width_mm = physical_width;
  output->physical_height_mm = physical_height;
  output->transform = transform;
}

static void wl_output_mode(void *data, struct wl_output *wl_output,
                           uint32_t flags, int32_t width, int32_t height,
                           int32_t refresh) {
  (void)wl_output;
  (void)refresh;
  if ((flags & WL_OUTPUT_MODE_CURRENT) != 0) {
    HudOutput *output = data;
    output->current_mode_width = width;
    output->current_mode_height = height;
    output->has_current_mode = true;
  }
}

static void wl_output_done(void *data, struct wl_output *wl_output) {
  (void)data;
  (void)wl_output;
}

static void wl_output_scale(void *data, struct wl_output *wl_output,
                            int32_t factor) {
  (void)data;
  (void)wl_output;
  (void)factor;
}

static const struct wl_output_listener WL_OUTPUT_LISTENER = {
    .geometry = wl_output_geometry,
    .mode = wl_output_mode,
    .done = wl_output_done,
    .scale = wl_output_scale,
};

static void output_logical_position(void *data,
                                    struct zxdg_output_v1 *xdg_output,
                                    int32_t x, int32_t y) {
  (void)xdg_output;
  HudOutput *output = data;
  output->x = x;
  output->y = y;
}

static void output_logical_size(void *data, struct zxdg_output_v1 *xdg_output,
                                int32_t width, int32_t height) {
  (void)xdg_output;
  HudOutput *output = data;
  output->width = width;
  output->height = height;
}

static void output_done(void *data, struct zxdg_output_v1 *xdg_output) {
  (void)data;
  (void)xdg_output;
}

static void output_name(void *data, struct zxdg_output_v1 *xdg_output,
                        const char *name) {
  (void)xdg_output;
  HudOutput *output = data;
  snprintf(output->name, sizeof(output->name), "%s", name);
}

static void output_description(void *data, struct zxdg_output_v1 *xdg_output,
                               const char *description) {
  (void)xdg_output;
  HudOutput *output = data;
  snprintf(output->description, sizeof(output->description), "%s", description);
}

static const struct zxdg_output_v1_listener OUTPUT_LISTENER = {
    .logical_position = output_logical_position,
    .logical_size = output_logical_size,
    .done = output_done,
    .name = output_name,
    .description = output_description,
};

static void registry_global(void *data, struct wl_registry *registry,
                            uint32_t name, const char *interface,
                            uint32_t version) {
  HudState *state = data;
  if (strcmp(interface, wl_compositor_interface.name) == 0) {
    uint32_t bind_version = version < 4 ? version : 4;
    state->compositor = wl_registry_bind(
        registry, name, &wl_compositor_interface, bind_version);
  } else if (strcmp(interface, wl_shm_interface.name) == 0) {
    state->shm = wl_registry_bind(registry, name, &wl_shm_interface, 1);
  } else if (strcmp(interface, zwlr_layer_shell_v1_interface.name) == 0) {
    uint32_t bind_version = version < 4 ? version : 4;
    state->layer_shell = wl_registry_bind(
        registry, name, &zwlr_layer_shell_v1_interface, bind_version);
  } else if (strcmp(interface, zxdg_output_manager_v1_interface.name) == 0) {
    uint32_t bind_version = version < 3 ? version : 3;
    state->output_manager = wl_registry_bind(
        registry, name, &zxdg_output_manager_v1_interface, bind_version);
  } else if (strcmp(interface, wl_output_interface.name) == 0 &&
             state->output_count < HUD_MAX_OUTPUTS) {
    HudOutput *output = &state->outputs[state->output_count++];
    output->registry_name = name;
    output->active = true;
    uint32_t bind_version = version < 3 ? version : 3;
    output->output =
        wl_registry_bind(registry, name, &wl_output_interface, bind_version);
    wl_output_add_listener(output->output, &WL_OUTPUT_LISTENER, output);
  }
}

static void registry_remove(void *data, struct wl_registry *registry,
                            uint32_t name) {
  (void)registry;
  HudState *state = data;
  for (size_t index = 0; index < state->output_count; index++) {
    if (state->outputs[index].registry_name == name) {
      state->outputs[index].active = false;
      return;
    }
  }
}

static const struct wl_registry_listener REGISTRY_LISTENER = {
    .global = registry_global,
    .global_remove = registry_remove,
};

static void layer_configure(void *data,
                            struct zwlr_layer_surface_v1 *layer_surface,
                            uint32_t serial, uint32_t width, uint32_t height) {
  (void)width;
  (void)height;
  HudState *state = data;
  zwlr_layer_surface_v1_ack_configure(layer_surface, serial);
  state->configured = true;
}

static void layer_closed(void *data,
                         struct zwlr_layer_surface_v1 *layer_surface) {
  (void)layer_surface;
  ((HudState *)data)->closed = true;
}

static const struct zwlr_layer_surface_v1_listener LAYER_LISTENER = {
    .configure = layer_configure,
    .closed = layer_closed,
};

static int create_buffers(HudState *state) {
  size_t frame_size = (size_t)state->frame_width * (size_t)state->frame_height *
                      sizeof(uint32_t);
  state->pool_size = HUD_BUFFER_COUNT * frame_size;
  int fd = anonymous_file(state->pool_size);
  if (fd < 0) {
    set_error(state, "cannot allocate Wayland renderer buffers");
    return -1;
  }
  state->pool_data =
      mmap(NULL, state->pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (state->pool_data == MAP_FAILED) {
    close(fd);
    state->pool_data = NULL;
    set_error(state, "cannot map Wayland renderer buffers");
    return -1;
  }
  state->pool = wl_shm_create_pool(state->shm, fd, (int32_t)state->pool_size);
  close(fd);
  for (int index = 0; index < HUD_BUFFER_COUNT; index++) {
    HudBuffer *frame = &state->buffers[index];
    frame->pixels = (uint32_t *)((unsigned char *)state->pool_data +
                                 (size_t)index * frame_size);
    frame->buffer = wl_shm_pool_create_buffer(
        state->pool, (int32_t)((size_t)index * frame_size), state->frame_width,
        state->frame_height, state->frame_width * 4, WL_SHM_FORMAT_ARGB8888);
    wl_buffer_add_listener(frame->buffer, &BUFFER_LISTENER, frame);
  }
  return 0;
}

static void rounded_rectangle(cairo_t *cairo, double x, double y, double width,
                              double height, double radius) {
  const double pi = 3.14159265358979323846;
  cairo_new_sub_path(cairo);
  cairo_arc(cairo, x + width - radius, y + radius, radius, -pi / 2.0, 0.0);
  cairo_arc(cairo, x + width - radius, y + height - radius, radius, 0.0,
            pi / 2.0);
  cairo_arc(cairo, x + radius, y + height - radius, radius, pi / 2.0, pi);
  cairo_arc(cairo, x + radius, y + radius, radius, pi, 3.0 * pi / 2.0);
  cairo_close_path(cairo);
}

static void render_hud(HudState *state, HudBuffer *frame, const char *text) {
  memset(frame->pixels, 0,
         (size_t)state->frame_width * (size_t)state->frame_height *
             sizeof(*frame->pixels));
  cairo_surface_t *surface = cairo_image_surface_create_for_data(
      (unsigned char *)frame->pixels, CAIRO_FORMAT_ARGB32, state->frame_width,
      state->frame_height, state->frame_width * 4);
  cairo_t *cairo = cairo_create(surface);

  cairo_set_operator(cairo, CAIRO_OPERATOR_SOURCE);
  cairo_set_source_rgba(cairo, 0.07, 0.07, 0.07, 0.74);
  rounded_rectangle(cairo, 1.0, 1.0, state->frame_width - 2.0,
                    state->frame_height - 2.0, HUD_RADIUS);
  cairo_fill_preserve(cairo);
  cairo_set_line_width(cairo, 1.0);
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 0.25);
  cairo_stroke(cairo);

  PangoLayout *layout = pango_cairo_create_layout(cairo);
  PangoFontDescription *font =
      pango_font_description_from_string("Monospace 13");
  pango_layout_set_font_description(layout, font);
  pango_layout_set_text(layout, text, -1);
  pango_layout_set_spacing(layout, 2 * PANGO_SCALE);
  cairo_move_to(cairo, 14.0, 12.0);
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 0.96);
  pango_cairo_show_layout(cairo, layout);

  pango_font_description_free(font);
  g_object_unref(layout);
  cairo_destroy(cairo);
  cairo_surface_flush(surface);
  cairo_surface_mark_dirty(surface);
  cairo_surface_destroy(surface);
}

static void render_training(HudState *state, HudBuffer *frame, double x,
                            double y, double diameter, const char *label) {
  memset(frame->pixels, 0,
         (size_t)state->frame_width * (size_t)state->frame_height *
             sizeof(*frame->pixels));
  cairo_surface_t *surface = cairo_image_surface_create_for_data(
      (unsigned char *)frame->pixels, CAIRO_FORMAT_ARGB32, state->frame_width,
      state->frame_height, state->frame_width * 4);
  cairo_t *cairo = cairo_create(surface);

  const double pi = 3.14159265358979323846;
  cairo_set_operator(cairo, CAIRO_OPERATOR_SOURCE);
  cairo_set_source_rgba(cairo, 0.92, 0.04, 0.04, 0.84);
  cairo_arc(cairo, x, y, diameter / 2.0, 0.0, 2.0 * pi);
  cairo_fill_preserve(cairo);
  cairo_set_line_width(cairo, 4.0);
  cairo_set_source_rgba(cairo, 1.0, 0.78, 0.78, 0.98);
  cairo_stroke(cairo);

  PangoLayout *layout = pango_cairo_create_layout(cairo);
  PangoFontDescription *font =
      pango_font_description_from_string("Sans Bold 16");
  pango_layout_set_font_description(layout, font);
  pango_layout_set_text(layout, label, -1);
  int text_width = 0;
  int text_height = 0;
  pango_layout_get_pixel_size(layout, &text_width, &text_height);
  cairo_set_source_rgba(cairo, 0.04, 0.04, 0.04, 0.78);
  rounded_rectangle(cairo, 18.0, 18.0, text_width + 24.0, text_height + 18.0,
                    HUD_RADIUS);
  cairo_fill(cairo);
  cairo_move_to(cairo, 30.0, 27.0);
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 0.98);
  pango_cairo_show_layout(cairo, layout);

  pango_font_description_free(font);
  g_object_unref(layout);
  cairo_destroy(cairo);
  cairo_surface_flush(surface);
  cairo_surface_mark_dirty(surface);
  cairo_surface_destroy(surface);
}

static void render_refinement_grid(HudState *state, HudBuffer *frame,
                                   double left, double top, double width,
                                   double height, int32_t depth,
                                   const char *source, int32_t row_count,
                                   int32_t column_count, const char *labels) {
  memset(frame->pixels, 0,
         (size_t)state->frame_width * (size_t)state->frame_height *
             sizeof(*frame->pixels));
  cairo_surface_t *surface = cairo_image_surface_create_for_data(
      (unsigned char *)frame->pixels, CAIRO_FORMAT_ARGB32, state->frame_width,
      state->frame_height, state->frame_width * 4);
  cairo_t *cairo = cairo_create(surface);
  cairo_set_operator(cairo, CAIRO_OPERATOR_SOURCE);
  cairo_set_line_width(cairo, 5.0);
  cairo_set_source_rgba(cairo, 0.0, 0.95, 1.0, 0.92);
  cairo_rectangle(cairo, left, top, width, height);
  cairo_stroke(cairo);
  cairo_set_line_width(cairo, 3.0);
  for (int32_t index = 1; index < column_count; ++index) {
    cairo_move_to(cairo, left + width * index / column_count, top);
    cairo_line_to(cairo, left + width * index / column_count, top + height);
  }
  for (int32_t index = 1; index < row_count; ++index) {
    cairo_move_to(cairo, left, top + height * index / row_count);
    cairo_line_to(cairo, left + width, top + height * index / row_count);
  }
  cairo_stroke(cairo);

  PangoLayout *label_layout = pango_cairo_create_layout(cairo);
  PangoFontDescription *label_font =
      pango_font_description_from_string("Sans Bold 30");
  pango_layout_set_font_description(label_layout, label_font);
  for (int32_t row = 0; row < row_count; ++row) {
    for (int32_t column = 0; column < column_count; ++column) {
      double center_x = left + width * (column + 0.5) / column_count;
      double center_y = top + height * (row + 0.5) / row_count;
      if (center_x < 0.0 || center_x >= state->frame_width || center_y < 0.0 ||
          center_y >= state->frame_height) {
        continue;
      }
      char label[2] = {labels[row * column_count + column], '\0'};
      pango_layout_set_text(label_layout, label, 1);
      int text_width = 0;
      int text_height = 0;
      pango_layout_get_pixel_size(label_layout, &text_width, &text_height);
      cairo_set_source_rgba(cairo, 0.02, 0.02, 0.02, 0.72);
      cairo_arc(cairo, center_x, center_y, 24.0, 0.0, 6.28318530718);
      cairo_fill(cairo);
      cairo_move_to(cairo, center_x - text_width / 2.0,
                    center_y - text_height / 2.0);
      cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 1.0);
      pango_cairo_show_layout(cairo, label_layout);
    }
  }

  PangoLayout *status = pango_cairo_create_layout(cairo);
  PangoFontDescription *status_font =
      pango_font_description_from_string("Sans Bold 18");
  pango_layout_set_font_description(status, status_font);
  char status_text[256];
  snprintf(status_text, sizeof(status_text), "Refinement %dx%d depth %d — %s",
           row_count, column_count, depth, source);
  pango_layout_set_text(status, status_text, -1);
  int status_width = 0;
  int status_height = 0;
  pango_layout_get_pixel_size(status, &status_width, &status_height);
  cairo_set_source_rgba(cairo, 0.02, 0.02, 0.02, 0.82);
  rounded_rectangle(cairo, 18.0, 18.0, status_width + 24.0,
                    status_height + 18.0, HUD_RADIUS);
  cairo_fill(cairo);
  cairo_move_to(cairo, 30.0, 27.0);
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 0.98);
  pango_cairo_show_layout(cairo, status);

  pango_font_description_free(label_font);
  pango_font_description_free(status_font);
  g_object_unref(label_layout);
  g_object_unref(status);
  cairo_destroy(cairo);
  cairo_surface_flush(surface);
  cairo_surface_mark_dirty(surface);
  cairo_surface_destroy(surface);
}

static void render_training_message(HudState *state, HudBuffer *frame,
                                    const char *text) {
  memset(frame->pixels, 0,
         (size_t)state->frame_width * (size_t)state->frame_height *
             sizeof(*frame->pixels));
  cairo_surface_t *surface = cairo_image_surface_create_for_data(
      (unsigned char *)frame->pixels, CAIRO_FORMAT_ARGB32, state->frame_width,
      state->frame_height, state->frame_width * 4);
  cairo_t *cairo = cairo_create(surface);
  PangoLayout *layout = pango_cairo_create_layout(cairo);
  PangoFontDescription *font =
      pango_font_description_from_string("Sans Bold 28");
  pango_layout_set_font_description(layout, font);
  pango_layout_set_text(layout, text, -1);
  double panel_width = state->frame_width - 128.0;
  if (panel_width > 900.0) {
    panel_width = 900.0;
  }
  pango_layout_set_width(layout, (int)((panel_width - 80.0) * PANGO_SCALE));
  pango_layout_set_wrap(layout, PANGO_WRAP_WORD_CHAR);
  pango_layout_set_alignment(layout, PANGO_ALIGN_CENTER);
  int text_width = 0;
  int text_height = 0;
  pango_layout_get_pixel_size(layout, &text_width, &text_height);
  (void)text_width;
  double panel_height = text_height + 56.0;
  double panel_x = (state->frame_width - panel_width) / 2.0;
  double panel_y = (state->frame_height - panel_height) / 2.0;

  cairo_set_operator(cairo, CAIRO_OPERATOR_SOURCE);
  cairo_set_source_rgba(cairo, 0.04, 0.04, 0.04, 0.90);
  rounded_rectangle(cairo, panel_x, panel_y, panel_width, panel_height, 16.0);
  cairo_fill_preserve(cairo);
  cairo_set_line_width(cairo, 2.0);
  cairo_set_source_rgba(cairo, 0.92, 0.04, 0.04, 0.95);
  cairo_stroke(cairo);
  cairo_move_to(cairo, panel_x + 40.0, panel_y + 28.0);
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 0.98);
  pango_cairo_show_layout(cairo, layout);

  pango_font_description_free(font);
  g_object_unref(layout);
  cairo_destroy(cairo);
  cairo_surface_flush(surface);
  cairo_surface_mark_dirty(surface);
  cairo_surface_destroy(surface);
}

static void render_training_cue(HudState *state, HudBuffer *frame,
                                const char *direction, double next_x,
                                double next_y, double next_diameter,
                                double prior_x, double prior_y,
                                double prior_diameter, double prior_opacity,
                                const char *label) {
  memset(frame->pixels, 0,
         (size_t)state->frame_width * (size_t)state->frame_height *
             sizeof(*frame->pixels));
  cairo_surface_t *surface = cairo_image_surface_create_for_data(
      (unsigned char *)frame->pixels, CAIRO_FORMAT_ARGB32, state->frame_width,
      state->frame_height, state->frame_width * 4);
  cairo_t *cairo = cairo_create(surface);
  const double pi = 3.14159265358979323846;

  if (prior_diameter > 0.0 && prior_opacity > 0.0) {
    cairo_set_operator(cairo, CAIRO_OPERATOR_SOURCE);
    cairo_set_source_rgba(cairo, 0.92, 0.04, 0.04, 0.84 * prior_opacity);
    cairo_arc(cairo, prior_x, prior_y, prior_diameter / 2.0, 0.0, 2.0 * pi);
    cairo_fill_preserve(cairo);
    cairo_set_line_width(cairo, 4.0);
    cairo_set_source_rgba(cairo, 1.0, 0.78, 0.78, 0.98 * prior_opacity);
    cairo_stroke(cairo);
  }

  if (next_diameter > 0.0) {
    cairo_set_operator(cairo, CAIRO_OPERATOR_OVER);
    cairo_set_source_rgba(cairo, 0.92, 0.04, 0.04, 0.68);
    cairo_arc(cairo, next_x, next_y, next_diameter / 2.0, 0.0, 2.0 * pi);
    cairo_fill_preserve(cairo);
    cairo_set_line_width(cairo, 4.0);
    cairo_set_source_rgba(cairo, 1.0, 0.78, 0.78, 0.98);
    cairo_stroke(cairo);
    double square_left = next_x - next_diameter / 2.0;
    double square_top = next_y - next_diameter / 2.0;
    double square_right = next_x + next_diameter / 2.0;
    double square_bottom = next_y + next_diameter / 2.0;
    cairo_set_line_width(cairo, 9.0);
    cairo_set_source_rgba(cairo, 0.0, 0.0, 0.0, 0.98);
    cairo_rectangle(cairo, square_left, square_top, next_diameter,
                    next_diameter);
    cairo_stroke(cairo);
    cairo_set_line_width(cairo, 5.0);
    cairo_set_source_rgba(cairo, 1.0, 0.92, 0.0, 1.0);
    cairo_move_to(cairo, square_left, square_top);
    cairo_line_to(cairo, square_right, square_top);
    cairo_stroke(cairo);
    cairo_set_source_rgba(cairo, 0.0, 0.90, 1.0, 1.0);
    cairo_move_to(cairo, square_right, square_top);
    cairo_line_to(cairo, square_right, square_bottom);
    cairo_stroke(cairo);
    cairo_set_source_rgba(cairo, 1.0, 0.12, 0.78, 1.0);
    cairo_move_to(cairo, square_right, square_bottom);
    cairo_line_to(cairo, square_left, square_bottom);
    cairo_stroke(cairo);
    cairo_set_source_rgba(cairo, 0.30, 1.0, 0.18, 1.0);
    cairo_move_to(cairo, square_left, square_bottom);
    cairo_line_to(cairo, square_left, square_top);
    cairo_stroke(cairo);
  }

  bool target_here = next_diameter > 0.0;
  double panel_width = state->frame_width - 96.0;
  if (panel_width > 560.0) {
    panel_width = 560.0;
  }
  double panel_height = target_here ? 120.0 : 280.0;
  if (panel_height > state->frame_height - 96.0) {
    panel_height = state->frame_height - 96.0;
  }
  double panel_x = (state->frame_width - panel_width) / 2.0;
  double panel_y = (state->frame_height - panel_height) / 2.0;
  if (target_here) {
    panel_y = next_y < state->frame_height / 2.0
                  ? state->frame_height - panel_height - 32.0
                  : 32.0;
  }
  cairo_set_operator(cairo, CAIRO_OPERATOR_OVER);
  cairo_set_source_rgba(cairo, 0.04, 0.04, 0.04, 0.94);
  rounded_rectangle(cairo, panel_x, panel_y, panel_width, panel_height, 18.0);
  cairo_fill_preserve(cairo);
  cairo_set_line_width(cairo, 3.0);
  cairo_set_source_rgba(cairo, 0.92, 0.04, 0.04, 0.98);
  cairo_stroke(cairo);

  PangoLayout *arrow = pango_cairo_create_layout(cairo);
  PangoFontDescription *arrow_font = pango_font_description_from_string(
      target_here ? "Sans Bold 42" : "Sans Bold 96");
  pango_layout_set_font_description(arrow, arrow_font);
  pango_layout_set_width(arrow, (int)((panel_width - 48.0) * PANGO_SCALE));
  pango_layout_set_alignment(arrow, PANGO_ALIGN_CENTER);
  pango_layout_set_text(arrow, direction, -1);
  cairo_move_to(cairo, panel_x + 24.0, panel_y + (target_here ? 4.0 : 18.0));
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 1.0);
  pango_cairo_show_layout(cairo, arrow);

  PangoLayout *text = pango_cairo_create_layout(cairo);
  PangoFontDescription *text_font =
      pango_font_description_from_string("Sans Bold 22");
  pango_layout_set_font_description(text, text_font);
  pango_layout_set_width(text, (int)((panel_width - 48.0) * PANGO_SCALE));
  pango_layout_set_wrap(text, PANGO_WRAP_WORD_CHAR);
  pango_layout_set_alignment(text, PANGO_ALIGN_CENTER);
  pango_layout_set_text(text, label, -1);
  cairo_move_to(cairo, panel_x + 24.0, panel_y + panel_height - 48.0);
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 1.0);
  pango_cairo_show_layout(cairo, text);

  pango_font_description_free(arrow_font);
  pango_font_description_free(text_font);
  g_object_unref(arrow);
  g_object_unref(text);
  cairo_destroy(cairo);
  cairo_surface_flush(surface);
  cairo_surface_mark_dirty(surface);
  cairo_surface_destroy(surface);
}

static void render_head_diagnostic(HudState *state, HudBuffer *frame,
                                   const uint8_t *pixels, int32_t source_width,
                                   int32_t source_height, int32_t source_stride,
                                   double bounds_x, double bounds_y,
                                   double bounds_width, double bounds_height,
                                   double pitch, double yaw, double roll,
                                   bool has_bounds, bool has_pose,
                                   const char *label) {
  memset(frame->pixels, 0,
         (size_t)state->frame_width * (size_t)state->frame_height *
             sizeof(*frame->pixels));
  double available_width = state->frame_width - 96.0;
  double available_height = state->frame_height - 220.0;
  double scale_x = available_width / source_width;
  double scale_y = available_height / source_height;
  double scale = scale_x < scale_y ? scale_x : scale_y;
  if (scale > 1.5) {
    scale = 1.5;
  }
  int32_t image_width = (int32_t)(source_width * scale);
  int32_t image_height = (int32_t)(source_height * scale);
  int32_t image_x = (state->frame_width - image_width) / 2;
  int32_t image_y = 48;
  for (int32_t destination_y = 0; destination_y < image_height;
       destination_y++) {
    int32_t source_y = (int32_t)(destination_y / scale);
    const uint8_t *source_row = pixels + (size_t)source_y * source_stride;
    uint32_t *destination =
        frame->pixels + (size_t)(image_y + destination_y) * state->frame_width +
        image_x;
    for (int32_t destination_x = 0; destination_x < image_width;
         destination_x++) {
      int32_t source_x = (int32_t)(destination_x / scale);
      const uint8_t *bgr = source_row + (size_t)source_x * 3;
      destination[destination_x] = 0xff000000u | ((uint32_t)bgr[2] << 16) |
                                   ((uint32_t)bgr[1] << 8) | bgr[0];
    }
  }

  cairo_surface_t *surface = cairo_image_surface_create_for_data(
      (unsigned char *)frame->pixels, CAIRO_FORMAT_ARGB32, state->frame_width,
      state->frame_height, state->frame_width * 4);
  cairo_t *cairo = cairo_create(surface);
  cairo_set_line_width(cairo, 5.0);
  cairo_set_source_rgba(cairo, 0.0, 0.0, 0.0, 0.95);
  cairo_rectangle(cairo, image_x - 3.0, image_y - 3.0, image_width + 6.0,
                  image_height + 6.0);
  cairo_stroke(cairo);
  if (has_bounds) {
    double x = image_x + bounds_x * image_width;
    double y = image_y + bounds_y * image_height;
    double width = bounds_width * image_width;
    double height = bounds_height * image_height;
    cairo_set_line_width(cairo, 4.0);
    cairo_set_source_rgba(cairo, 0.25, 1.0, 0.2, 1.0);
    cairo_rectangle(cairo, x, y, width, height);
    cairo_stroke(cairo);
    if (has_pose) {
      double center_x = x + width / 2.0;
      double center_y = y + height / 2.0;
      double axis = width < height ? width * 0.35 : height * 0.35;
      cairo_set_line_width(cairo, 5.0);
      cairo_set_source_rgba(cairo, 1.0, 0.2, 0.2, 1.0);
      cairo_move_to(cairo, center_x, center_y);
      cairo_line_to(cairo, center_x + axis * cos(yaw * 0.01745329252),
                    center_y + axis * sin(roll * 0.01745329252));
      cairo_stroke(cairo);
      cairo_set_source_rgba(cairo, 0.2, 0.7, 1.0, 1.0);
      cairo_move_to(cairo, center_x, center_y);
      cairo_line_to(cairo, center_x + axis * sin(yaw * 0.01745329252),
                    center_y + axis * sin(pitch * 0.01745329252));
      cairo_stroke(cairo);
    }
  }

  PangoLayout *layout = pango_cairo_create_layout(cairo);
  PangoFontDescription *font =
      pango_font_description_from_string("Sans Bold 22");
  double maximum_panel_width = state->frame_width - 96.0;
  if (maximum_panel_width > 720.0) {
    maximum_panel_width = 720.0;
  }
  pango_layout_set_font_description(layout, font);
  pango_layout_set_width(layout,
                         (int)((maximum_panel_width - 48.0) * PANGO_SCALE));
  pango_layout_set_wrap(layout, PANGO_WRAP_WORD_CHAR);
  pango_layout_set_alignment(layout, PANGO_ALIGN_CENTER);
  pango_layout_set_text(layout, label, -1);
  int text_width = 0;
  int text_height = 0;
  pango_layout_get_pixel_size(layout, &text_width, &text_height);
  double panel_width = text_width + 48.0;
  if (panel_width < 320.0) {
    panel_width = 320.0;
  }
  if (panel_width > maximum_panel_width) {
    panel_width = maximum_panel_width;
  }
  double panel_height = text_height + 40.0;
  if (panel_height < 80.0) {
    panel_height = 80.0;
  }
  double panel_x = (state->frame_width - panel_width) / 2.0;
  double panel_y = image_y + image_height + 24.0;
  if (panel_y + panel_height > state->frame_height - 24.0) {
    panel_y = state->frame_height - panel_height - 24.0;
  }
  pango_layout_set_width(layout, (int)((panel_width - 48.0) * PANGO_SCALE));
  cairo_set_source_rgba(cairo, 0.03, 0.03, 0.03, 0.94);
  rounded_rectangle(cairo, panel_x, panel_y, panel_width, panel_height, 16.0);
  cairo_fill_preserve(cairo);
  cairo_set_line_width(cairo, 3.0);
  cairo_set_source_rgba(cairo, 1.0, 0.75, 0.0, 1.0);
  cairo_stroke(cairo);
  cairo_move_to(cairo, panel_x + 24.0,
                panel_y + (panel_height - text_height) / 2.0);
  cairo_set_source_rgba(cairo, 1.0, 1.0, 1.0, 1.0);
  pango_cairo_show_layout(cairo, layout);
  pango_font_description_free(font);
  g_object_unref(layout);
  cairo_destroy(cairo);
  cairo_surface_flush(surface);
  cairo_surface_mark_dirty(surface);
  cairo_surface_destroy(surface);
}

static HudBuffer *available_buffer(HudState *state) {
  for (int attempt = 0; attempt < HUD_BUFFER_COUNT; attempt++) {
    int candidate = (state->next_buffer + attempt) % HUD_BUFFER_COUNT;
    if (!state->buffers[candidate].busy) {
      state->next_buffer = (candidate + 1) % HUD_BUFFER_COUNT;
      return &state->buffers[candidate];
    }
  }
  return NULL;
}

static void cleanup(HudState *state) {
  if (state == NULL) {
    return;
  }
  for (int index = 0; index < HUD_BUFFER_COUNT; index++) {
    if (state->buffers[index].buffer != NULL) {
      wl_buffer_destroy(state->buffers[index].buffer);
    }
  }
  for (size_t index = 0; index < state->output_count; index++) {
    if (state->outputs[index].xdg_output != NULL) {
      zxdg_output_v1_destroy(state->outputs[index].xdg_output);
    }
    if (state->outputs[index].output != NULL) {
      wl_output_destroy(state->outputs[index].output);
    }
  }
  if (state->layer_surface != NULL) {
    zwlr_layer_surface_v1_destroy(state->layer_surface);
  }
  if (state->surface != NULL) {
    wl_surface_destroy(state->surface);
  }
  if (state->pool != NULL) {
    wl_shm_pool_destroy(state->pool);
  }
  if (state->pool_data != NULL) {
    munmap(state->pool_data, state->pool_size);
  }
  if (state->output_manager != NULL) {
    zxdg_output_manager_v1_destroy(state->output_manager);
  }
  if (state->layer_shell != NULL) {
    zwlr_layer_shell_v1_destroy(state->layer_shell);
  }
  if (state->shm != NULL) {
    wl_shm_destroy(state->shm);
  }
  if (state->compositor != NULL) {
    wl_compositor_destroy(state->compositor);
  }
  if (state->registry != NULL) {
    wl_registry_destroy(state->registry);
  }
  if (state->display != NULL) {
    wl_display_disconnect(state->display);
  }
  free(state);
}

static HudState *create_wayland_state(char *error, size_t error_size) {
  HudState *state = calloc(1, sizeof(*state));
  if (state == NULL) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "cannot allocate Wayland renderer state");
    }
    return NULL;
  }
  state->display = wl_display_connect(NULL);
  if (state->display == NULL) {
    set_error(state, "cannot connect renderer to the Wayland display");
  }
  if (!state->failed) {
    state->registry = wl_display_get_registry(state->display);
    wl_registry_add_listener(state->registry, &REGISTRY_LISTENER, state);
    if (wl_display_roundtrip(state->display) < 0) {
      set_error(state, "Wayland disconnected while starting renderer");
    }
  }
  if (!state->failed &&
      (state->compositor == NULL || state->shm == NULL ||
       state->layer_shell == NULL || state->output_manager == NULL)) {
    set_error(
        state,
        "Wayland compositor does not provide layer shell and output geometry");
  }
  if (!state->failed) {
    for (size_t index = 0; index < state->output_count; index++) {
      HudOutput *output = &state->outputs[index];
      output->xdg_output = zxdg_output_manager_v1_get_xdg_output(
          state->output_manager, output->output);
      zxdg_output_v1_add_listener(output->xdg_output, &OUTPUT_LISTENER, output);
    }
    if (wl_display_roundtrip(state->display) < 0) {
      set_error(state, "Wayland disconnected while reading output geometry");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    cleanup(state);
    return NULL;
  }
  return state;
}

__attribute__((visibility("default"))) void *
gazeebo_hud_create(char *error, size_t error_size) {
  HudState *state = create_wayland_state(error, error_size);
  if (state == NULL) {
    return NULL;
  }
  state->frame_width = HUD_WIDTH;
  state->frame_height = HUD_HEIGHT;
  if (!state->failed && create_buffers(state) != 0) {
    state->failed = true;
  }
  if (!state->failed) {
    state->surface = wl_compositor_create_surface(state->compositor);
    state->layer_surface = zwlr_layer_shell_v1_get_layer_surface(
        state->layer_shell, state->surface, NULL,
        ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY, "gazeebo-debug-hud");
    if (state->surface == NULL || state->layer_surface == NULL) {
      set_error(state, "cannot create debug HUD layer surface");
    }
  }
  if (!state->failed) {
    zwlr_layer_surface_v1_add_listener(state->layer_surface, &LAYER_LISTENER,
                                       state);
    zwlr_layer_surface_v1_set_size(state->layer_surface, HUD_WIDTH, HUD_HEIGHT);
    zwlr_layer_surface_v1_set_anchor(state->layer_surface,
                                     ZWLR_LAYER_SURFACE_V1_ANCHOR_TOP |
                                         ZWLR_LAYER_SURFACE_V1_ANCHOR_RIGHT);
    zwlr_layer_surface_v1_set_margin(state->layer_surface, HUD_MARGIN,
                                     HUD_MARGIN, 0, 0);
    zwlr_layer_surface_v1_set_exclusive_zone(state->layer_surface, -1);
    zwlr_layer_surface_v1_set_keyboard_interactivity(
        state->layer_surface,
        ZWLR_LAYER_SURFACE_V1_KEYBOARD_INTERACTIVITY_NONE);
    struct wl_region *empty = wl_compositor_create_region(state->compositor);
    wl_surface_set_input_region(state->surface, empty);
    wl_region_destroy(empty);
    wl_surface_commit(state->surface);
    while (!state->configured && !state->closed && !state->failed) {
      if (wl_display_dispatch(state->display) < 0) {
        set_error(state, "Wayland disconnected while configuring debug HUD");
      }
    }
  }
  if (state->failed || state->closed) {
    copy_error(state, error, error_size);
    cleanup(state);
    return NULL;
  }
  return state;
}

static HudOutput *output_at(HudState *state, double x, double y) {
  for (size_t index = 0; index < state->output_count; index++) {
    HudOutput *output = &state->outputs[index];
    if (output->active && output->width > 0 && output->height > 0 &&
        y >= output->y && x < output->x + output->width &&
        y < output->y + output->height) {
      return output;
    }
  }
  return NULL;
}

static HudOutput *exact_output(HudState *state, int32_t x, int32_t y,
                               int32_t width, int32_t height) {
  HudOutput *match = NULL;
  for (size_t index = 0; index < state->output_count; index++) {
    HudOutput *output = &state->outputs[index];
    if (output->active && output->x == x && output->y == y &&
        output->width == width && output->height == height) {
      if (match != NULL) {
        return NULL;
      }
      match = output;
    }
  }
  return match;
}

static void format_hud_text(HudState *state, const char *region_id, double x,
                            double y, char *text, size_t text_size) {
  HudOutput *output = output_at(state, x, y);
  const char *description = output != NULL && output->description[0] != '\0'
                                ? output->description
                                : "unknown output";
  const char *name =
      output != NULL && output->name[0] != '\0' ? output->name : "unknown";
  snprintf(text, text_size,
           "output: %s\nconnector: %s\nregion: %s\nx: %.0f  y: %.0f",
           description, name, region_id, x, y);
}

__attribute__((visibility("default"))) int
gazeebo_hud_update(void *handle, const char *region_id, double x, double y,
                   char *error, size_t error_size) {
  HudState *state = handle;
  if (state == NULL || region_id == NULL || state->closed) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "debug HUD is closed");
    }
    return -1;
  }
  (void)wl_display_dispatch_pending(state->display);
  HudBuffer *frame = available_buffer(state);
  if (frame == NULL) {
    if (wl_display_roundtrip(state->display) < 0) {
      set_error(state, "Wayland disconnected while updating debug HUD");
    }
    frame = available_buffer(state);
  }
  if (frame == NULL) {
    set_error(state, "debug HUD has no available drawing buffer");
  }
  if (!state->failed) {
    char text[1024];
    format_hud_text(state, region_id, x, y, text, sizeof(text));
    render_hud(state, frame, text);
    frame->busy = true;
    wl_surface_attach(state->surface, frame->buffer, 0, 0);
    wl_surface_damage_buffer(state->surface, 0, 0, state->frame_width,
                             state->frame_height);
    wl_surface_commit(state->surface);
    if (wl_display_flush(state->display) < 0 && errno != EAGAIN) {
      set_error(state, "cannot flush debug HUD update");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) void gazeebo_hud_destroy(void *handle) {
  cleanup(handle);
}

__attribute__((visibility("default"))) void *
gazeebo_training_create(int32_t region_x, int32_t region_y,
                        int32_t region_width, int32_t region_height,
                        char *error, size_t error_size) {
  if (region_width <= 0 || region_height <= 0) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration training region is invalid");
    }
    return NULL;
  }
  HudState *state = create_wayland_state(error, error_size);
  if (state == NULL) {
    return NULL;
  }
  HudOutput *output = output_at(state, region_x + region_width / 2.0,
                                region_y + region_height / 2.0);
  if (output == NULL) {
    set_error(state, "selected portal region does not match a Wayland output");
  }
  state->training_output = output;
  state->training_metrics_exact =
      output != NULL && exact_output(state, region_x, region_y, region_width,
                                     region_height) == output;
  state->frame_width = region_width;
  state->frame_height = region_height;
  if (!state->failed && create_buffers(state) != 0) {
    state->failed = true;
  }
  if (!state->failed) {
    state->surface = wl_compositor_create_surface(state->compositor);
    state->layer_surface = zwlr_layer_shell_v1_get_layer_surface(
        state->layer_shell, state->surface, output->output,
        ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY, TRAINING_NAMESPACE);
    if (state->surface == NULL || state->layer_surface == NULL) {
      set_error(state, "cannot create calibration training layer surface");
    }
  }
  if (!state->failed) {
    zwlr_layer_surface_v1_add_listener(state->layer_surface, &LAYER_LISTENER,
                                       state);
    zwlr_layer_surface_v1_set_size(state->layer_surface, region_width,
                                   region_height);
    zwlr_layer_surface_v1_set_anchor(state->layer_surface,
                                     ZWLR_LAYER_SURFACE_V1_ANCHOR_TOP |
                                         ZWLR_LAYER_SURFACE_V1_ANCHOR_RIGHT |
                                         ZWLR_LAYER_SURFACE_V1_ANCHOR_BOTTOM |
                                         ZWLR_LAYER_SURFACE_V1_ANCHOR_LEFT);
    zwlr_layer_surface_v1_set_exclusive_zone(state->layer_surface, -1);
    zwlr_layer_surface_v1_set_keyboard_interactivity(
        state->layer_surface,
        ZWLR_LAYER_SURFACE_V1_KEYBOARD_INTERACTIVITY_NONE);
    struct wl_region *empty = wl_compositor_create_region(state->compositor);
    wl_surface_set_input_region(state->surface, empty);
    wl_region_destroy(empty);
    wl_surface_commit(state->surface);
    while (!state->configured && !state->closed && !state->failed) {
      if (wl_display_dispatch(state->display) < 0) {
        set_error(
            state,
            "Wayland disconnected while configuring calibration training");
      }
    }
  }
  if (state->failed || state->closed) {
    copy_error(state, error, error_size);
    cleanup(state);
    return NULL;
  }
  return state;
}

__attribute__((visibility("default"))) int
gazeebo_training_show_target(void *handle, double x, double y, double diameter,
                             const char *label, char *error,
                             size_t error_size) {
  HudState *state = handle;
  if (state == NULL || label == NULL || state->closed) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration training is closed");
    }
    return -1;
  }
  if (diameter <= 0.0 || x - diameter / 2.0 < -0.01 ||
      y - diameter / 2.0 < -0.01 ||
      x + diameter / 2.0 > state->frame_width + 0.01 ||
      y + diameter / 2.0 > state->frame_height + 0.01) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration target is outside its display");
    }
    return -1;
  }
  (void)wl_display_dispatch_pending(state->display);
  HudBuffer *frame = available_buffer(state);
  if (frame == NULL) {
    if (wl_display_roundtrip(state->display) < 0) {
      set_error(state,
                "Wayland disconnected while updating calibration training");
    }
    frame = available_buffer(state);
  }
  if (frame == NULL) {
    set_error(state, "calibration training has no available drawing buffer");
  }
  if (!state->failed) {
    render_training(state, frame, x, y, diameter, label);
    frame->busy = true;
    wl_surface_attach(state->surface, frame->buffer, 0, 0);
    wl_surface_damage_buffer(state->surface, 0, 0, state->frame_width,
                             state->frame_height);
    wl_surface_commit(state->surface);
    if (wl_display_flush(state->display) < 0 && errno != EAGAIN) {
      set_error(state, "cannot flush calibration training update");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) int
gazeebo_training_show_grid(void *handle, double left, double top, double width,
                           double height, int32_t depth, const char *source,
                           int32_t row_count, int32_t column_count,
                           const char *labels, char *error, size_t error_size) {
  HudState *state = handle;
  if (state == NULL || source == NULL || labels == NULL || state->closed ||
      width <= 0.0 || height <= 0.0 || depth < 0 || row_count < 2 ||
      row_count > 6 || column_count < 2 || column_count > 6 ||
      strlen(labels) != (size_t)(row_count * column_count)) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "refinement grid input is invalid");
    }
    return -1;
  }
  (void)wl_display_dispatch_pending(state->display);
  HudBuffer *frame = available_buffer(state);
  if (frame == NULL && wl_display_roundtrip(state->display) >= 0) {
    frame = available_buffer(state);
  }
  if (frame == NULL) {
    set_error(state, "refinement grid has no available drawing buffer");
  }
  if (!state->failed) {
    render_refinement_grid(state, frame, left, top, width, height, depth,
                           source, row_count, column_count, labels);
    frame->busy = true;
    wl_surface_attach(state->surface, frame->buffer, 0, 0);
    wl_surface_damage_buffer(state->surface, 0, 0, state->frame_width,
                             state->frame_height);
    wl_surface_commit(state->surface);
    if (wl_display_flush(state->display) < 0 && errno != EAGAIN) {
      set_error(state, "cannot flush refinement grid update");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) int
gazeebo_training_show_message(void *handle, const char *label, char *error,
                              size_t error_size) {
  HudState *state = handle;
  if (state == NULL || label == NULL || state->closed) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration training is closed");
    }
    return -1;
  }
  (void)wl_display_dispatch_pending(state->display);
  HudBuffer *frame = available_buffer(state);
  if (frame == NULL && wl_display_roundtrip(state->display) >= 0) {
    frame = available_buffer(state);
  }
  if (frame == NULL) {
    set_error(state, "calibration training has no available drawing buffer");
  }
  if (!state->failed) {
    render_training_message(state, frame, label);
    frame->busy = true;
    wl_surface_attach(state->surface, frame->buffer, 0, 0);
    wl_surface_damage_buffer(state->surface, 0, 0, state->frame_width,
                             state->frame_height);
    wl_surface_commit(state->surface);
    if (wl_display_flush(state->display) < 0 && errno != EAGAIN) {
      set_error(state, "cannot flush calibration training message");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) int gazeebo_training_show_cue(
    void *handle, const char *direction, double next_x, double next_y,
    double next_diameter, double prior_x, double prior_y, double prior_diameter,
    double prior_opacity, const char *label, char *error, size_t error_size) {
  HudState *state = handle;
  if (state == NULL || direction == NULL || label == NULL || state->closed) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration training is closed");
    }
    return -1;
  }
  if (prior_opacity < 0.0 || prior_opacity > 1.0) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration cue opacity is invalid");
    }
    return -1;
  }
  (void)wl_display_dispatch_pending(state->display);
  HudBuffer *frame = available_buffer(state);
  if (frame == NULL && wl_display_roundtrip(state->display) >= 0) {
    frame = available_buffer(state);
  }
  if (frame == NULL) {
    set_error(state, "calibration training has no available cue buffer");
  }
  if (!state->failed) {
    render_training_cue(state, frame, direction, next_x, next_y, next_diameter,
                        prior_x, prior_y, prior_diameter, prior_opacity, label);
    frame->busy = true;
    wl_surface_attach(state->surface, frame->buffer, 0, 0);
    wl_surface_damage_buffer(state->surface, 0, 0, state->frame_width,
                             state->frame_height);
    wl_surface_commit(state->surface);
    if (wl_display_flush(state->display) < 0 && errno != EAGAIN) {
      set_error(state, "cannot flush calibration training cue");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) int gazeebo_training_show_diagnostic(
    void *handle, const uint8_t *pixels, int32_t source_width,
    int32_t source_height, int32_t source_stride, double bounds_x,
    double bounds_y, double bounds_width, double bounds_height, double pitch,
    double yaw, double roll, int32_t has_bounds, int32_t has_pose,
    const char *label, char *error, size_t error_size) {
  HudState *state = handle;
  if (state == NULL || pixels == NULL || label == NULL || state->closed ||
      source_width <= 0 || source_height <= 0 ||
      source_stride < source_width * 3) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "head diagnostic input is invalid");
    }
    return -1;
  }
  (void)wl_display_dispatch_pending(state->display);
  HudBuffer *frame = available_buffer(state);
  if (frame == NULL && wl_display_roundtrip(state->display) >= 0) {
    frame = available_buffer(state);
  }
  if (frame == NULL) {
    set_error(state, "head diagnostic has no available drawing buffer");
  }
  if (!state->failed) {
    render_head_diagnostic(state, frame, pixels, source_width, source_height,
                           source_stride, bounds_x, bounds_y, bounds_width,
                           bounds_height, pitch, yaw, roll, has_bounds != 0,
                           has_pose != 0, label);
    frame->busy = true;
    wl_surface_attach(state->surface, frame->buffer, 0, 0);
    wl_surface_damage_buffer(state->surface, 0, 0, state->frame_width,
                             state->frame_height);
    wl_surface_commit(state->surface);
    if (wl_display_flush(state->display) < 0 && errno != EAGAIN) {
      set_error(state, "cannot flush head diagnostic update");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) int
gazeebo_training_hide(void *handle, char *error, size_t error_size) {
  HudState *state = handle;
  if (state == NULL || state->closed) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "calibration training is closed");
    }
    return -1;
  }
  (void)wl_display_dispatch_pending(state->display);
  HudBuffer *frame = available_buffer(state);
  if (frame == NULL && wl_display_roundtrip(state->display) >= 0) {
    frame = available_buffer(state);
  }
  if (frame == NULL) {
    set_error(state, "calibration training has no available clearing buffer");
  }
  if (!state->failed) {
    memset(frame->pixels, 0,
           (size_t)state->frame_width * (size_t)state->frame_height *
               sizeof(*frame->pixels));
    frame->busy = true;
    wl_surface_attach(state->surface, frame->buffer, 0, 0);
    wl_surface_damage_buffer(state->surface, 0, 0, state->frame_width,
                             state->frame_height);
    wl_surface_commit(state->surface);
    if (wl_display_flush(state->display) < 0 && errno != EAGAIN) {
      set_error(state, "cannot flush calibration training clear");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) int gazeebo_training_display_metrics(
    void *handle, int32_t *mode_width, int32_t *mode_height,
    int32_t *physical_width_mm, int32_t *physical_height_mm) {
  HudState *state = handle;
  if (state == NULL || mode_width == NULL || mode_height == NULL ||
      physical_width_mm == NULL || physical_height_mm == NULL ||
      !state->training_metrics_exact || state->training_output == NULL ||
      !state->training_output->has_current_mode ||
      state->training_output->current_mode_width <= 0 ||
      state->training_output->current_mode_height <= 0 ||
      state->training_output->physical_width_mm <= 0 ||
      state->training_output->physical_height_mm <= 0) {
    return -1;
  }
  HudOutput *output = state->training_output;
  bool rotated = output->transform == WL_OUTPUT_TRANSFORM_90 ||
                 output->transform == WL_OUTPUT_TRANSFORM_270 ||
                 output->transform == WL_OUTPUT_TRANSFORM_FLIPPED_90 ||
                 output->transform == WL_OUTPUT_TRANSFORM_FLIPPED_270;
  *mode_width =
      rotated ? output->current_mode_height : output->current_mode_width;
  *mode_height =
      rotated ? output->current_mode_width : output->current_mode_height;
  *physical_width_mm =
      rotated ? output->physical_height_mm : output->physical_width_mm;
  *physical_height_mm =
      rotated ? output->physical_width_mm : output->physical_height_mm;
  return 0;
}

__attribute__((visibility("default"))) void
gazeebo_training_destroy(void *handle) {
  cleanup(handle);
}

__attribute__((visibility("default"))) void *
gazeebo_display_monitor_create(char *error, size_t error_size) {
  return create_wayland_state(error, error_size);
}

__attribute__((visibility("default"))) int
gazeebo_display_monitor_snapshot(void *handle, char *snapshot,
                                 size_t snapshot_size, char *error,
                                 size_t error_size) {
  HudState *state = handle;
  if (state == NULL || snapshot == NULL || snapshot_size == 0 ||
      state->closed) {
    if (error != NULL && error_size > 0) {
      snprintf(error, error_size, "display monitor is closed");
    }
    return -1;
  }
  if (wl_display_roundtrip(state->display) < 0) {
    set_error(state, "Wayland disconnected while refreshing outputs");
  }
  bool attached = false;
  if (!state->failed) {
    for (size_t index = 0; index < state->output_count; index++) {
      HudOutput *output = &state->outputs[index];
      if (output->active && output->xdg_output == NULL) {
        output->xdg_output = zxdg_output_manager_v1_get_xdg_output(
            state->output_manager, output->output);
        zxdg_output_v1_add_listener(output->xdg_output, &OUTPUT_LISTENER,
                                    output);
        attached = true;
      }
    }
  }
  if (attached && wl_display_roundtrip(state->display) < 0) {
    set_error(state, "Wayland disconnected while reading new output geometry");
  }
  size_t used = 0;
  size_t count = 0;
  if (!state->failed) {
    snapshot[0] = '\0';
    for (size_t index = 0; index < state->output_count; index++) {
      HudOutput *output = &state->outputs[index];
      if (!output->active || output->width <= 0 || output->height <= 0) {
        continue;
      }
      int written = snprintf(snapshot + used, snapshot_size - used,
                             "%s%d:%d:%d:%d", count == 0 ? "" : ";", output->x,
                             output->y, output->width, output->height);
      if (written < 0 || (size_t)written >= snapshot_size - used) {
        set_error(state, "display topology snapshot is too large");
        break;
      }
      used += (size_t)written;
      count++;
    }
    if (count == 0) {
      set_error(state, "display monitor found no active outputs");
    }
  }
  if (state->failed) {
    copy_error(state, error, error_size);
    return -1;
  }
  return 0;
}

__attribute__((visibility("default"))) void
gazeebo_display_monitor_destroy(void *handle) {
  cleanup(handle);
}
