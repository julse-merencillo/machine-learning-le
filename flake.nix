# flake.nix
{
  description = "Traffic RL with SUMO";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python313;
        
        pythonEnv = python.withPackages (ps: with ps; [
          numpy
          gymnasium
          matplotlib
          pandas
          torch
	  pip
          tensorboard
          jupyter
          # Add others as needed
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.sumo
          ];

          shellHook = ''
            export SUMO_HOME="${pkgs.sumo}/share/sumo"
            export PATH="${pkgs.sumo}/bin:$PATH"
            
            # Install packages not in nixpkgs
            export PIP_PREFIX="$(pwd)/.pip_packages"
            export PYTHONPATH="$PIP_PREFIX/${python.sitePackages}:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"
            
            pip install --quiet traci sumolib stable-baselines3 "sumo-rl>=1.4" 2>/dev/null
            
            echo "🚦 Traffic RL environment loaded"
          '';
        };
      }
    );
}
