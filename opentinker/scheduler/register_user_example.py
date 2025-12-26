#!/usr/bin/env python3
"""
Example: User Registration and Authentication

This script demonstrates how to register a user and use the API key
for authenticated scheduler operations.
"""

import requests

SCHEDULER_URL = "http://localhost:8765"


def main():
    print("=" * 60)
    print("User Registration Example")
    print("=" * 60)

    # Step 1: Register a new user
    username = input("Enter username to register: ")

    print(f"\n📝 Registering user '{username}'...")

    response = requests.post(f"{SCHEDULER_URL}/register", params={"username": username})

    if response.status_code == 200:
        result = response.json()
        print("\n✅ Registration successful!")
        print("=" * 60)
        print("🔑 YOUR API KEY (save this - cannot be retrieved later!):")
        print("")
        print(f"  {result['api_key']}")
        print("")
        print("=" * 60)
        print(f"User ID: {result['user_id']}")
        print(f"Username: {result['username']}")

        # Step 2: Test authentication with the API key
        api_key = result["api_key"]
        print("\n✅ Testing authentication...")

        # Try to list jobs with the API key
        headers = {"Authorization": f"Bearer {api_key}"}
        jobs_response = requests.get(f"{SCHEDULER_URL}/list_jobs", headers=headers)

        if jobs_response.status_code == 200:
            print("✅ Authentication successful!")
            jobs = jobs_response.json()
            print(f"Current jobs: {len(jobs['jobs'])}")
        else:
            print(f"❌ Failed to list jobs: {jobs_response.text}")

        # Save to file for easy reference
        with open(f".api_key_{username}", "w") as f:
            f.write(api_key)
        print(f"\n💾 API key saved to .api_key_{username}")

    else:
        print(f"❌ Registration failed: {response.text}")


if __name__ == "__main__":
    main()
