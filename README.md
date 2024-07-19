## Mosquito Alert Data Visualization Dashboard

This dashboard visualizes Mosquito Alert (MA) reports, monitoring performance, coverage, and user retention. The primary goal is to provide daily updated time-series data filtered by geographic, administrative, and bio-climatic areas.

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

### Installation

To set up the environment, follow these steps:

1. Get the bootstrap makefile and execute it in your shell:

    ```shell
    $ make -f bootstrap run-setup
    ```

    This command installs necessary dependencies (`git`, `git-lfs`, and `micromamba`) and clones the repository from GitHub. You will need to provide your GitHub credentials (username and API key) during this process. After cloning, the makefile sets up a virtual environment and decrypts secret files upon password input. Once completed, the application is ready to be launched.

2. Depending on your intended use, configure the `env` variable in `config.ini` to either *dev* (development) or *prod* (production). In case of a development environment, it is necessary to uncomment in the `env.yaml` packages relative to *DevOps* before installing the conda environment.

3. To ensure the application functions correctly, data must first be retrieved from the `tigadata` database (i.e., data ingestion) and the server must be started. You can manually perform these tasks using the following make commands:

    ```shell
    $ make -f makefile pipeline
    $ make -f makefile serve_local_access
    ```

By following these steps, you'll have the application set up and running according to your specified environment configuration.

### Set-up
The application configuration files are found in the `./config` folder. There are two types of environment configurations: production `./config/prod/*.ini` and development `./config/dev/*.ini`. Data is ingested from the **tigadata** database of MA. If running on the production server with the DB already running, set *conn_type* to *local* in the *params* section of `./config/config.ini` and provide *local_db* credentials in the `./config/prod/config.ini`. Avoid storing secrets in `config.ini` but use `.config.ini` instead, which is ignored by GitHub.

For ingesting data from a remote server, in `./config/config.ini` set *conn_type* to *remote* and provide *remote_db* and *ssh* credentials in `./config/prod/config.ini`. To send logs to Grafana-Loki and email alerts, supply API tokens in the *logging* section. Also, specify the app's absolute *root* path in the *paths* section.

```ini [title=".config.ini"]
[local_db]
    password = ***
[remote_db]
    password = ***
[ssh]
    # Server where the remote_db is running
    password = ***
[logging]
    # Loki cloud access policies token
    api_key_loki = ***
    # Google Account application API token
    api_key_gmail = ***
[paths]
    # Absolute root path of the application
    root = /home/user/project_path/
```
In order to access the dashboard, it is necessary to register the users and their passwords withing a `.credentials.json` setup file.

```json [title=".credentials.json"]
{
    "user_1": "user_1_password",
    "user_2": "user_2_password",
    "admin": "admin_password"
}
```
