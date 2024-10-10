import os
import glob
import subprocess
import argparse
import getpass
from typing import Optional


# Function to encrypt or decrypt a single file using OpenSSL or Sops
# NOTE: Could replace bash commands used in the makefile if OpenSSL is preferred
def process_file(
    input_file: str,
    output_file: str,
    password: str,
    method: str,
    engine: str,
    sops_age_key_file: Optional[str] = None,
) -> None:
    """
    This function encrypts or decrypts a single file using OpenSSL or Sops.
    It takes the following parameters:
    - input_file: The path to the input file.
    - output_file: The path to the output file.
    - password: The encryption/decryption password.
    - method: The method to use, either 'encrypt' or 'decrypt'.
    - engine: The encryption/decryption engine to use, either 'openssl' or 'sops'.
    - sops_age_key_file: The path to the Sops age key file, only required for Sops engine.
    """
    if engine == "openssl":
        openssl_cmd = f'openssl enc -aes-256-cbc -salt -in "{input_file}" -out "{output_file}" -k "{password}"'
        if method == "encrypt":
            openssl_cmd = openssl_cmd + " -pbkdf2"
        elif method == "decrypt":
            openssl_cmd = openssl_cmd + " -d -pbkdf2"
        else:
            print(f"Unsupported method: {method}. Use 'encrypt' or 'decrypt'.")
            return
    elif engine == "sops":
        if method == "encrypt":
            sops_cmd = f'sops -e "{input_file}" > "{output_file}"'
        elif method == "decrypt":
            if input_file.endswith(".json"):
                sops_cmd = f'SOPS_AGE_KEY_FILE={sops_age_key_file} sops --input-type json --output-type json -d "{input_file}" > "{output_file}"'
            elif input_file.endswith(".ini"):
                sops_cmd = f'SOPS_AGE_KEY_FILE={sops_age_key_file} sops --input-type ini --output-type ini -d "{input_file}" > "{output_file}"'
            else:
                print(
                    f"Unsupported file type for sops method: {input_file}. Use .json or .ini."
                )
                return
        else:
            print(f"Unsupported method: {method}. Use 'encrypt' or 'decrypt'.")
            return
    else:
        print(f"Unsupported engine: {engine}. Use 'openssl' or 'sops'.")
        KeyError()

    # Execute the command using subprocess
    try:
        if engine == "openssl":
            subprocess.run(openssl_cmd, shell=True, check=True)
            if method == "encrypt":
                print(f"Encrypted {input_file} successfully using openssl.")
            elif method == "decrypt":
                print(f"Decrypted {input_file} successfully using openssl.")
        elif engine == "sops":
            subprocess.run(sops_cmd, shell=True, check=True)
            if method == "encrypt":
                print(f"Encrypted {input_file} successfully using sops.")
            elif method == "decrypt":
                print(f"Decrypted {input_file} successfully using sops.")
    except subprocess.CalledProcessError as e:
        if engine == "openssl":
            if method == "encrypt":
                print(f"Failed to encrypt {input_file}. Error: {e}")
            elif method == "decrypt":
                print(f"Failed to decrypt {input_file}. Error: {e}")
        elif engine == "sops":
            if method == "encrypt":
                print(f"Failed to encrypt {input_file} using sops. Error: {e}")
            elif method == "decrypt":
                print(f"Failed to decrypt {input_file} using sops. Error: {e}")


# Function to process files in a directory using glob
def process_files_in_directory(
    directory: str, password: str, method: str, pattern: str = "*"
) -> None:
    """
    This function processes files in a directory using glob.
    It takes the following parameters:
    - directory: The directory to process.
    - password: The encryption/decryption password.
    - method: The method to use, either 'encrypt' or 'decrypt'.
    - pattern: The file name pattern to search for (default: all files).
    """
    # Use glob to find all files recursively
    files_to_process = glob.glob(os.path.join(directory, "**", pattern), recursive=True)

    for file_to_process in files_to_process:
        if method == "encrypt":
            if not file_to_process.endswith(".enc"):  # Exclude already encrypted files
                encrypted_file = file_to_process + ".enc"
                process_file(file_to_process, encrypted_file, password, method)
        elif method == "decrypt":
            if file_to_process.endswith(".enc"):  # Only decrypt .enc files
                decrypted_file = file_to_process[:-4]  # Remove .enc extension
                process_file(file_to_process, decrypted_file, password, method)


def parse_arguments() -> argparse.Namespace:
    """
    This function parses the command line arguments.
    It returns the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Encrypt or decrypt files recursively in a directory."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Directory to process",
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        help="Encryption/decryption password (optional)",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["encrypt", "decrypt"],
        required=True,
        help="Method: encrypt or decrypt",
    )
    parser.add_argument(
        "-fp",
        "--pattern",
        type=str,
        default="*",
        help="Pattern: file name pattern to search for (default: all files)",
    )

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    import os

    args = parse_arguments()

    # Validate directory argument
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory.")
        exit(1)
    if not args.password:
        password = getpass.getpass("Enter encryption/decryption password: ")
    else:
        password = args.password
    # Process files based on method
    process_files_in_directory(args.directory, password, args.method, args.pattern)
