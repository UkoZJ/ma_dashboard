# Mosquito Alert Data Visualization Dashboard

This dashboard visualizes Mosquito Alert (MA) reports, monitoring performance, coverage, and user retention. The primary goal is to provide daily updated time-series data filtered by geographic, administrative, and bio-climatic areas.

## Table of Contents

- [Mosquito Alert Data Visualization Dashboard](#mosquito-alert-data-visualization-dashboard)
  - [Table of Contents](#table-of-contents)
    - [Visualization Views](#visualization-views)
    - [Interaction and Filters](#interaction-and-filters)
  - [Project Setup](#project-setup)
  - [GIT Setup](#git-setup)
  - [Application Setup](#application-setup)
    - [Relevant Notebooks](#relevant-notebooks)
  - [Mosquito Suitability](#mosquito-suitability)
  - [Understanding Participation Patterns: Classifying MA-User Behavior](#understanding-participation-patterns-classifying-ma-user-behavior)
  - [Gravity Model input datasets](#gravity-model-input-datasets)

### Visualization Views

* **Overview**: Displays total report counts for the main report types of MA and citizen participation. Counts related to mosquitoes are grouped by spatial scale and presented in a table.
  
* **KPIs**: Presents key performance indexes to monitor data flow, monitoring performance, user retention, and sampling coverage of the MA system.
  
* **Filter**: Allows detailed exploration of mosquito-related reports.

### Interaction and Filters

The **Filter Parameters** control panel, located on the left side of the display, affects all views simultaneously. This panel allows filtering data by different spatial scales and time windows (from daily to yearly). 

* **Scale**: Sets the spatial scale over which to filter within the *Entity* parameter. There are four available scale types: [*GADM Levels*](https://gadm.org) that aims at mapping the administrative areas of all countries, at all levels of sub-division, [Köppen–Geiger's *Climatic Regions*](https://www.nature.com/articles/sdata2018214), *Biome* which classifies 14 distinct geographical region with specific climate, vegetation, and animal life and the [WWF's *Eco-regions*](https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world) that represent the original distribution of distinct assemblages of species and communities. Changing the *Scale* parameter directly affects the **Overview** table.
  
* **Entity**: Specify a geographic entity relative to the selected scale (default is `Total`, meaning all data are considered). In case *GADM Levels* are selected, administrative names are separated by a vertical pipe symbol. For example in case of GADM level 2 for `Barcelona | Cataluña | Spain`, `Barcelona` is a province (level 2), `Cataluña` is an autonomy (level 1) and `Spain` is a country name (level 0).
  
* **Frequency**: Determines the time window of data aggregation and it can be set to daily, weekly, monthly, quarterly and yearly.
  
* **Report Type**: Enables report filtering related to mosquito *adults*, *bites*, or *breeding sites*.

> [!NOTE]  
> Changes in parameters affect visualizations across different views, and it may not be immediately obvious since only one view is shown at a time.

## Project Setup

This project uses `make` commands for setup and management. Get the bootstrap makefile from the [project repository](https://github.com/Mosquito-Alert/expert_effort.git) and execute it in your shell:

```shell
$ make -f bootstrap setup
```

This command installs necessary dependencies (`git`, `micromamba`, etc.), clones the repository from GitHub and gets large-storage folders from GDrive. You will need to provide your GitHub credentials (username and API key) and login to your Google Account during this process. After cloning, the makefile sets up a virtual environment and decrypts secrets upon password input. Once completed, the application is ready to be launched.

> [!NOTE] 
> Check if `env-lock.yaml` (default) or `env.yaml` is used to build the environment. Remember that `env.yaml` will get the most recent versions of libraries which may break the code. Remember to rebuild  the `env-lock.yaml` with `conda-lock -f env.yaml -p linux-64` in case new libraries are added or removed to `env.yaml`. Sometimes conda-lock is not able to create a lock file (runs forever) and a combination of `micromamba env export > env-export.yaml` and `pip freeze > requirements.txt` is used instead. 

> [!NOTE]
> GDrive of `mosquitoalert2share@gmail.com` with link `https://drive.google.com/drive/u/1/folders/16kg55tSuqSrNxNaSMeYjeSIllB7M0IMK` contains the following compressed and password protected project folders: `.secrets` and `data.zip`.

## GIT Setup

Since there are configuration secrets in the project, it is necessary to install git hooks to prevent committing unencrypted files by running:

```shell
$ make install-git-hooks
```

Before pushing to repository it is necessary to encrypt sensitive files before pushing:

```shell
make encrypt
```

To push changes to remote repositories use:

```shell
make git-push message="Your commit message"
```

This encrypts sensitive files, commits changes, and pushes to configured remote repositories.

> [!CAUTION]
> Remember to keep your age key (`age-key.txt`) secure and share it only with trusted team members since this is the key used by SOPS to encrypt and decrypt secret config files.

> [!NOTE] 
> Do not push to the repository folders that of large size, but compress them and upload them in a dedicated GDrive project's folder.

## Application Setup

The application configuration files are found in the `./config` folder. There are two types of environment configurations: production `./config/prod/*.ini` and development `./config/dev/*.ini`. Data is ingested from the **tigadata** database of MA. If running on the production server with the DB already running, set *conn_type* to *local* in the *params* section of `./config/config.ini` and provide *local_db* credentials in the `./config/prod/config.ini`. Avoid storing secrets in `config.ini` but use `.config.ini` instead, which is ignored by GitHub.

For ingesting data from a remote server, in `./config/config.ini` set *conn_type* to *remote* and provide *remote_db* and *ssh* credentials in `./config/prod/config.ini`. To send logs to Grafana-Loki and email alerts, supply API tokens in the *logging* section. Also, specify the app's absolute *root* path in the *paths* section.

If the Tornado server of Panel is used to provide user access to the dashboard, it is necessary to register the users and their passwords withing a `.credentials.json` setup file.

To ensure the application functions correctly, data must first be retrieved from the `tigadata` database (i.e., data ingestion) and the server must be started. You can manually perform these tasks using the following make methods:

```shell
$ make -f makefile pipeline
$ make -f makefile serve_local_access
```

By following these steps, you'll have the application set up and running according to your specified environment configuration.


### Relevant Notebooks

## Mosquito Suitability

Worldwide mosquito suitability metrics are computed in the `mosquito_suitability.py` notebook. MA report density and mosquito suitability are joined in a composed index in order to assess the most relevant cell-region to sample from by CS reporting. Moreover, re-gridding on discrete global grids is investigated. The project is its early prototyping stage and further development is needed for production.

## Understanding Participation Patterns: Classifying MA-User Behavior

The `ppa.py` notebook is related to PhD chapter and paper of Ayat Abourashed. 

## Gravity Model input datasets

The `io_gravity_model.py` notebook provides input data for a gravity model developed at UAB by Dr. Daniel Campos Moreno.
