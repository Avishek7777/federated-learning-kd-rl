import subprocess

test_files = [
    "test_models.py",
    "test_data.py",
    "test_client_train.py",
    "test_distill.py"
]

print("Running all tests...\n")

for test_file in test_files:
    print(f"===== Running {test_file} =====")
    result = subprocess.run(["python", f"tests/{test_file}"])
    if result.returncode != 0:
        print(f"❌ {test_file} failed.\n")
        break
    else:
        print(f"✅ {test_file} passed.\n")

print("All tests finished.")
