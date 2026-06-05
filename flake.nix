{
  description = "Development environment for damacy";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    claude-code.url = "github:sadjow/claude-code-nix";
    claude-code.inputs.nixpkgs.follows = "nixpkgs";
    claude-code.inputs.flake-utils.follows = "flake-utils";
    codex-cli-nix.url = "github:sadjow/codex-cli-nix";
    codex-cli-nix.inputs.nixpkgs.follows = "nixpkgs";
    codex-cli-nix.inputs.flake-utils.follows = "flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      claude-code,
      codex-cli-nix,
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
            # ruff handles both lint and format for python; pyproject.toml
            # carries the per-project rule set.
            ruff = {
              enable = true;
            };
            ruff-format = {
              enable = true;
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
            codex-cli-nix.packages.${system}.default
            cmake
            ninja
            pkg-config
            git
            clang-tools
            # clang itself (not just clang-tools) is needed for the libFuzzer
            # build under tests/fuzz/. Ships libclang_rt.fuzzer alongside.
            clang
            uv
            tokei
            ruff
            pyright
          ];

          buildInputs = [
            cudaPkgs.cudatoolkit
            cudaPkgs.nvcomp
            cudaPkgs.nvcomp.static
            # libcufile is dlopen'd at runtime (see store_fs_gds.c):
            # cufile.h via CUFILE_ROOT, libcufile.so.0 via LD_LIBRARY_PATH.
            cudaPkgs.libcufile
            cudaPkgs.nsight_systems
            cudaPkgs.nsight_compute
          ] ++ (with pkgs; [
            # libnuma is loaded at runtime via dlopen (see src/numa/numa.c);
            # only needed if you want NUMA pinning to actually do anything.
            # Kept in the devShell so multi-socket dev boxes resolve a node.
            numactl
            lldb
            perf
            gh
            man-pages
            man-pages-posix
            liburing
          ]);

          CUDA_PATH = "${cudaPkgs.cudatoolkit}";
          CUDAToolkit_ROOT = "${cudaPkgs.cudatoolkit}";
          Nvcomp_ROOT = "${cudaPkgs.nvcomp}";
          CUFILE_ROOT = "${cudaPkgs.libcufile}";
          # Point cuFile at the vendored config (allow_compat_mode = true)
          # so cuFileDriverOpen succeeds on hosts without nvidia-fs. In
          # compat mode reads go through cuFile's host-bounce buffers
          # internally — slower than real GDS, but exercises the same
          # store_fs_gds code path.
          CUFILE_ENV_PATH_JSON = "${cudaPkgs.libcufile}/etc/cufile.json";

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
              pkgs.numactl  # libnuma for src/numa
              # cuFile dlopens libmount.so (util-linux) and libudev.so
              # (systemd) at driver init even in compat mode.
              pkgs.util-linux.lib
              pkgs.systemd
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
