import os
import glob
import subprocess
import argparse
import getpass


# Function to encrypt or decrypt a single file using OpenSSL
def process_file(input_file, output_file, password, method):

    openssl_cmd = f'openssl enc -aes-256-cbc -salt -in "{input_file}" -out "{output_file}" -k "{password}"'
    if method == "encrypt":
        openssl_cmd = openssl_cmd + " -pbkdf2"
    elif method == "decrypt":
        # Construct the OpenSSL command for decryption
        openssl_cmd = openssl_cmd + " -d -pbkdf2"
    else:
        print(f"Unsupported method: {method}. Use 'encrypt' or 'decrypt'.")
        return

    # Execute the command using subprocess
    try:
        subprocess.run(openssl_cmd, shell=True, check=True)
        if method == "encrypt":
            print(f"Encrypted {input_file} successfully.")
        elif method == "decrypt":
            print(f"Decrypted {input_file} successfully.")
    except subprocess.CalledProcessError as e:
        if method == "encrypt":
            print(f"Failed to encrypt {input_file}. Error: {e}")
        elif method == "decrypt":
            print(f"Failed to decrypt {input_file}. Error: {e}")


# Function to process files in a directory using glob
def process_files_in_directory(directory, password, method, pattern="*"):
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


def parse_arguments():
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
