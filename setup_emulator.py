#!/usr/bin/env python3
"""One-time emulator setup script for Android World.

Installs all required apps and creates snapshots on the emulator.
Usage: python setup_emulator.py [--console_port 5556]
"""
import os
import sys
import argparse
import traceback

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = 'none'

def main():
    parser = argparse.ArgumentParser(description="Setup Android World emulator")
    parser.add_argument("--console_port", type=int, default=5556)
    parser.add_argument("--adb_path", type=str, default=None)
    parser.add_argument("--skip_apps", nargs="*", default=[], help="App names to skip (e.g., joplin)")
    args = parser.parse_args()

    adb_path = args.adb_path or os.environ.get("ADB_PATH") or "adb"
    
    import shutil
    if not shutil.which(adb_path):
        print(f"ERROR: adb not found at '{adb_path}'")
        sys.exit(1)

    print(f"Setting up emulator on console port {args.console_port}...")
    print(f"Using adb: {adb_path}")
    if args.skip_apps:
        print(f"Skipping apps: {args.skip_apps}")

    from android_world.env import env_launcher
    from android_world.env.setup_device import setup
    
    # Load env WITHOUT running setup (emulator_setup=False)
    print("Loading environment...")
    sys.stdout.flush()
    env = env_launcher.load_and_setup_env(
        console_port=args.console_port,
        emulator_setup=False,
        adb_path=adb_path,
    )
    
    # Run setup app-by-app with error handling
    print(f"\nInstalling {len(setup._APPS)} apps (skipping failures)...")
    sys.stdout.flush()
    
    skip_set = {s.lower() for s in args.skip_apps}
    succeeded = []
    failed = []
    skipped = []
    
    from android_world.env import adb_utils
    adb_utils.press_home_button(env.controller)
    adb_utils.set_root_if_needed(env.controller)

    for app_cls in setup._APPS:
        app_name = app_cls.app_name
        if app_name.lower() in skip_set:
            print(f"  SKIP  {app_name}")
            skipped.append(app_name)
            continue
        try:
            # Step 1: Download and install APK (for third-party apps)
            print(f"  Installing APK for {app_name}...", end=" ", flush=True)
            setup.maybe_install_app(app_cls, env)
            print("done.", end=" ", flush=True)

            # Step 2: Setup app (clear data, copy files, create snapshot)
            print(f"Setting up...", end=" ", flush=True)
            setup.setup_app(app_cls, env)
            print("OK")
            succeeded.append(app_name)
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((app_name, str(e)))
    
    print(f"\n{'='*60}")
    print(f"Setup Summary:")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")
    print(f"  Skipped:   {len(skipped)}")
    if failed:
        print(f"\nFailed apps:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
    
    env.close()
    print("Done.")

if __name__ == "__main__":
    main()
