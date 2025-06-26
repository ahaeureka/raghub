from loguru import logger
from raghub_core.config.generator import TomlConfigGenerator
from raghub_core.utils.file.project import ProjectHelper
from raghub_interfaces.config.interface_config import InerfaceConfig

if __name__ == "__main__":
    toml_content, markdown_content = TomlConfigGenerator.generate_toml_config(InerfaceConfig)
    with open(f"{ProjectHelper.get_project_root().as_posix()}/docs/config.md", "w") as f:
        f.write(markdown_content)
    with open(f"{ProjectHelper.get_project_root().as_posix()}/configs/config.toml", "w") as f:
        f.write(toml_content)
    logger.debug("TOML and Markdown configuration files generated successfully.")
