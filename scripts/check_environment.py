from importlib.metadata import version, PackageNotFoundError


PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
]


def main() -> None:
    print("AI Development environment check")
    print("-" * 40)

    for package in PACKAGES:
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: not installed")


if __name__ == "__main__":
    main()