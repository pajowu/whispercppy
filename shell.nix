with import <nixpkgs> { };

pkgs.mkShell {
  buildInputs = with pkgs; [
    cmake
    git
    python39Packages.python
    python39Packages.pybind11
    python39Packages.setuptools
  ] ++

  # accelerates whisper.cpp on M{1,2} Macs
  (if !stdenv.isDarwin then [ ] else [
    darwin.apple_sdk.frameworks.Accelerate
  ]);
  CPPFLAGS = builtins.concatStringsSep " " [
    # (lib.removeSuffix "\n" (builtins.readFile "${pkgs.clang}/nix-support/cc-cflags"))
    # (lib.removeSuffix "\n" (builtins.readFile "${pkgs.clang}/nix-support/libc-cflags"))
    (lib.removeSuffix "\n" (builtins.readFile "${pkgs.clang}/nix-support/libcxx-cxxflags"))
  ];
  shellHook = "export CPPFLAGS+=$NIX_CFLAGS_COMPILE";


  # # use the system bazel (necessary on NixOS and Guix, as the downloaded bazel binary cannot run on these)
  # shellHook = ''
  #   export DISABLE_BAZEL_WRAPPER=1
  # '' + (
  #   pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
  #     # Use accellerate framework on darwin
  #     export BAZEL_LINKOPTS="-F ${pkgs.darwin.apple_sdk.frameworks.Accelerate}/Library/Frameworks"
  #   ''
  # );

}
