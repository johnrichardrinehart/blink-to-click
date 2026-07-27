{ pkgs, revision }:
let
  python = pkgs.python3;
  gazeebo = python.pkgs.buildPythonApplication {
    pname = "gazeebo";
    version = "0.1.0";
    pyproject = true;

    src = pkgs.lib.fileset.toSource {
      root = ../..;
      fileset = pkgs.lib.fileset.unions [
        ../../README.md
        ../../pyproject.toml
        ../../src
        ../../tests
      ];
    };

    build-system = [ python.pkgs.setuptools ];
    nativeBuildInputs = [
      pkgs.pkg-config
      pkgs.wayland-protocols
      pkgs.wayland-scanner
    ];
    buildInputs = [
      pkgs.cairo
      pkgs.libei
      pkgs.pango
      pkgs.wayland
    ];
    dependencies = with python.pkgs; [
      dbus-next
      numpy
      onnxruntime
      opencv4
    ];

    postInstall = ''
      hudBuildDir=$(mktemp -d)
      hudProtocol=${pkgs.wlr-protocols}/share/wlr-protocols/unstable/wlr-layer-shell-unstable-v1.xml
      xdgProtocol=${pkgs.wayland-protocols}/share/wayland-protocols/stable/xdg-shell/xdg-shell.xml
      outputProtocol=${pkgs.wayland-protocols}/share/wayland-protocols/unstable/xdg-output/xdg-output-unstable-v1.xml
      wayland-scanner client-header "$hudProtocol" \
        "$hudBuildDir/wlr-layer-shell-unstable-v1-client-protocol.h"
      wayland-scanner private-code "$hudProtocol" \
        "$hudBuildDir/wlr-layer-shell-unstable-v1-protocol.c"
      wayland-scanner private-code "$xdgProtocol" \
        "$hudBuildDir/xdg-shell-protocol.c"
      wayland-scanner client-header "$outputProtocol" \
        "$hudBuildDir/xdg-output-unstable-v1-client-protocol.h"
      wayland-scanner private-code "$outputProtocol" \
        "$hudBuildDir/xdg-output-unstable-v1-protocol.c"
      $CC -shared -fPIC -O2 -Wall -Wextra -Werror \
        -I"$hudBuildDir" \
        src/hud_native.c \
        src/input_native.c \
        "$hudBuildDir/wlr-layer-shell-unstable-v1-protocol.c" \
        "$hudBuildDir/xdg-shell-protocol.c" \
        "$hudBuildDir/xdg-output-unstable-v1-protocol.c" \
        -o "$hudBuildDir/libgazeebo-hud.so" \
        $(pkg-config --cflags --libs wayland-client pangocairo libei-1.0)
      install -Dm644 "$hudBuildDir/libgazeebo-hud.so" \
        "$out/lib/libgazeebo-hud.so"
    '';

    pythonImportsCheck = [
      "gazeebo"
      "gazeebo.adaptation"
      "gazeebo.calibration"
      "gazeebo.camera"
      "gazeebo.cli"
      "gazeebo.contexts"
      "gazeebo.contracts"
      "gazeebo.control"
      "gazeebo.display"
      "gazeebo.diagnostics"
      "gazeebo.training"
      "gazeebo.geometry"
      "gazeebo.hud"
      "gazeebo.input_capture"
      "gazeebo.native"
      "gazeebo.portal"
      "gazeebo.recovery"
      "gazeebo.runtime"
      "gazeebo.state"
      "gazeebo.vision"
    ];
    makeWrapperArgs = [
      "--unset"
      "PYTHONPATH"
      "--set"
      "GAZEEBO_BUILD_ID"
      revision
      "--set"
      "GAZEEBO_MODEL_DIR"
      "${pkgs.openseeface}/share/openseeface/models"
      "--set"
      "GAZEEBO_TRACKER_DIR"
      "${pkgs.openseeface}/share/openseeface"
      "--set"
      "GAZEEBO_HUD_LIBRARY"
      "${placeholder "out"}/lib/libgazeebo-hud.so"
      "--prefix"
      "PYTHONPATH"
      ":"
      "${pkgs.openseeface}/share/openseeface"
    ];
    doCheck = true;
    checkPhase = ''
      runHook preCheck
      export GAZEEBO_MODEL_DIR=${pkgs.openseeface}/share/openseeface/models
      export GAZEEBO_TRACKER_DIR=${pkgs.openseeface}/share/openseeface
      python - <<'PY'
      from gazeebo.vision import OpenSeeFaceEstimator

      estimator = OpenSeeFaceEstimator(640, 480)
      estimator.close()
      PY
      python -m unittest discover -s tests -v
      runHook postCheck
    '';

    meta = {
      description = "Process-scoped gaze-driven cursor navigation";
      license = pkgs.lib.licenses.mit;
      mainProgram = "gazeebo";
      platforms = [ "x86_64-linux" ];
    };
  };
in
{
  inherit gazeebo;
  default = gazeebo;
}
