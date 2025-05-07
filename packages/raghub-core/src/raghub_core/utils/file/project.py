from pathlib import Path

import tomli


class ProjectHelper:
    @staticmethod
    def get_project_name(project_path: str = "pyproject.toml") -> str:
        pyproject_path = Path(project_path)
        with open(pyproject_path, "rb") as f:
            data = tomli.load(f)
        return data["project"]["name"]

    @staticmethod
    def get_project_root() -> Path:
        """
        Find the root directory of the project by looking for the 'pyproject.toml' file.
        :return: Path to the project root directory.
        :raises FileNotFoundError: If the project root is not found.
        """
        current_path = Path(__file__).resolve()
        for parent in current_path.parents:
            if (parent / "pyproject.toml").exists() and ProjectHelper.get_project_name(
                (parent / "pyproject.toml").as_posix()
            ) == "raghub":
                return parent
        raise FileNotFoundError("Project root not found. Ensure you are within an elibris project.")
