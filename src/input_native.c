#include <libei.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_ERROR_SIZE 512

typedef void (*motion_callback)(int32_t absolute, double x, double y,
                                void *user_data);

typedef struct {
  struct ei *context;
} InputState;

static void copy_error(char *error, size_t error_size, const char *message) {
  if (error != NULL && error_size > 0) {
    snprintf(error, error_size, "%s", message);
  }
}

__attribute__((visibility("default"))) void *
gazeebo_input_create(int fd, char *error, size_t error_size) {
  if (fd < 0) {
    copy_error(error, error_size, "input-capture descriptor is invalid");
    return NULL;
  }
  InputState *state = calloc(1, sizeof(*state));
  if (state == NULL) {
    copy_error(error, error_size, "cannot allocate input-capture state");
    return NULL;
  }
  state->context = ei_new_receiver(NULL);
  if (state->context == NULL) {
    copy_error(error, error_size, "cannot create libei receiver");
    free(state);
    return NULL;
  }
  if (ei_setup_backend_fd(state->context, fd) != 0) {
    copy_error(error, error_size, "cannot connect libei receiver");
    ei_unref(state->context);
    free(state);
    return NULL;
  }
  return state;
}

__attribute__((visibility("default"))) int gazeebo_input_get_fd(void *handle) {
  InputState *state = handle;
  return state == NULL ? -1 : ei_get_fd(state->context);
}

__attribute__((visibility("default"))) int
gazeebo_input_dispatch(void *handle, motion_callback callback, void *user_data,
                       char *error, size_t error_size) {
  InputState *state = handle;
  if (state == NULL || callback == NULL) {
    copy_error(error, error_size, "input-capture dispatch state is invalid");
    return -1;
  }
  ei_dispatch(state->context);
  struct ei_event *event = NULL;
  while ((event = ei_get_event(state->context)) != NULL) {
    switch (ei_event_get_type(event)) {
    case EI_EVENT_SEAT_ADDED:
      ei_seat_bind_capabilities(ei_event_get_seat(event), EI_DEVICE_CAP_POINTER,
                                EI_DEVICE_CAP_POINTER_ABSOLUTE, NULL);
      break;
    case EI_EVENT_POINTER_MOTION:
      callback(0, ei_event_pointer_get_dx(event),
               ei_event_pointer_get_dy(event), user_data);
      break;
    case EI_EVENT_POINTER_MOTION_ABSOLUTE:
      callback(1, ei_event_pointer_get_absolute_x(event),
               ei_event_pointer_get_absolute_y(event), user_data);
      break;
    case EI_EVENT_DISCONNECT:
      ei_event_unref(event);
      copy_error(error, error_size, "libei input service disconnected");
      return -1;
    default:
      break;
    }
    ei_event_unref(event);
  }
  return 0;
}

__attribute__((visibility("default"))) void
gazeebo_input_destroy(void *handle) {
  InputState *state = handle;
  if (state == NULL) {
    return;
  }
  ei_unref(state->context);
  free(state);
}
