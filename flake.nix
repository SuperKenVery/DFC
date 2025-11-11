# SPDX-License-Identifier: Unlicense
{
  inputs = {
    # nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    # systems.url = "github:nix-systems/default";
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      nixgl,
      ...
    }:
    flake-utils.lib.eachSystem nixpkgs.lib.systems.flakeExposed (
      system:
      let
        pkgs = import nixpkgs { inherit system; overlays = [ nixgl.overlay ]; };
      in
      {
        devShells.default = (pkgs.buildFHSEnv {
          name = "new glibc";
          targetPkgs = (pkgs: with pkgs; [
            gcc
            glib
          ]);
          runScript = "${pkgs.fish}/bin/fish";
        }).env;
      }
    );
}
