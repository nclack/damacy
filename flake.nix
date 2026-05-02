{
  description = "Development environment for damacy";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    claude-code.url = "github:sadjow/claude-code-nix";
    claude-code.inputs.nixpkgs.follows = "nixpkgs";
    claude-code.inputs.flake-utils.follows = "flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      claude-code,
      git-hooks,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        cudaPkgs = pkgs.cudaPackages_13;
        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            clang-format = {
              enable = true;
              types_or = pkgs.lib.mkForce [
                "c"
                "c++"
                "cuda"
              ];
            };
            gersemi = {
              enable = true;
              name = "gersemi";
              entry = "${pkgs.gersemi}/bin/gersemi -i";
              files = "(\\.cmake$|CMakeLists\\.txt$)";
              pass_filenames = true;
            };
          };
        };
      in
      {
        checks = {
          inherit pre-commit-check;
        };

        formatter = pkgs.nixfmt-tree;

        devShells.default = pkgs.mkShell.override { stdenv = cudaPkgs.backendStdenv; } {
          name = "damacy";
          inherit (pre-commit-check) shellHook;

          nativeBuildInputs = with pkgs; [
            claude-code.packages.${system}.default
            cmake
            ninja
            pkg-config
            git
            clang-tools
            uv
            python311
          ];

          buildInputs = [
            cudaPkgs.cudatoolkit
            cudaPkgs.nvcomp
            cudaPkgs.nvcomp.static
            cudaPkgs.libcufile
          ] ++ (with pkgs; [
            lldb
            gh
            man-pages
            man-pages-posix
          ]);

          CUDA_PATH = "${cudaPkgs.cudatoolkit}";
          CUDAToolkit_ROOT = "${cudaPkgs.cudatoolkit}";
          Nvcomp_ROOT = "${cudaPkgs.nvcomp}";
          CUFILE_ROOT = "${cudaPkgs.libcufile}";

          # cudatoolkit ships stub libcuda.so.1; the real one lives with the
          # NVIDIA driver. Prepend the driver dir so the runtime resolves the
          # actual libcuda before the stub. /run/opengl-driver/lib is NixOS's
          # canonical driver lib dir (set by hardware.nvidia or the wrapper).
          LD_LIBRARY_PATH =
            "/run/opengl-driver/lib:"
            + pkgs.lib.makeLibraryPath [
              cudaPkgs.cudatoolkit
              "${cudaPkgs.cudatoolkit}/lib64"
              cudaPkgs.nvcomp
              cudaPkgs.libcufile
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib  # numpy wheel _multiarray_umath dlopens libz.so.1
            ];

          # uv defaults to ~/.cache/uv; pin a project-local cache so a `nix
          # develop` exit doesn't leave wheels in the user dir under nix-store
          # paths that may be GC'd. Keeps the bench env reproducible per-tree.
          UV_PROJECT_ENVIRONMENT = ".venv";

          # NixOS can't exec uv's pre-built dynamic-linked python (no generic
          # ld-linux). Force uv to use the nix-provided python311 instead.
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON = "python3.11";
        };
      }
    );
}
