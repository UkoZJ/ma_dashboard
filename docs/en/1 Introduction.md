<style>
.note-block {
    background-color: #1e3a5f;
    border-left: 6px solid #3399ff;
    padding: 10px;
    margin: 20px 0;
    color: #eeeeee;
    display: flex;
    align-items: center;
}
.note-block p {
    margin: 0;
}
.note-block strong {
    color: #66b2ff;
}
</style>

<h1 style="font-size:16px">Introduction</h1>

This dashboard visualizes Mosquito Alert (MA) reports, monitoring performance, coverage, and user retention. The primary goal is to provide daily updated time-series data filtered by geographic, administrative, and bio-climatic areas.

<h2 style="font-size:14px"> Visualization Views </h2>

* **Overview**: Displays total report counts for the main report types of MA and citizen participation. Counts related to mosquitoes are grouped by spatial scale and presented in a table.
* **KPIs**: Presents key performance indexes to monitor data flow, monitoring performance, user retention, and sampling coverage of the MA system.
* **Filter**: Allows detailed exploration of mosquito-related reports.

<h2 style="font-size:14px"> Filter Parameters </h2>

The **Filter Parameters** control panel, located on the left side of the display, affects all views simultaneously. This panel allows filtering data by different spatial scales and time windows (from daily to yearly). 

* **Scale**: Sets the spatial scale over which to filter within the *Entity* parameter. There are four available scale types: [*GADM Levels*](https://gadm.org) that aims at mapping the administrative areas of all countries, at all levels of sub-division, [Köppen–Geiger's *Climatic Regions*](https://www.nature.com/articles/sdata2018214), *Biome* which classifies 14 distinct geographical region with specific climate, vegetation, and animal life and the [WWF's *Eco-regions*](https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world) that represent the original distribution of distinct assemblages of species and communities. Changing the *Scale* parameter directly affects the **Overview** table.
* **Entity**: Specify a geographic entity relative to the selected scale (default is `Total`, meaning all data are considered). In case *GADM Levels* are selected, administrative names are separated by a vertical pipe symbol. For example in case of GADM level 2 for `Barcelona | Cataluña | Spain`, `Barcelona` is a province (level 2), `Cataluña` is an autonomy (level 1) and `Spain` is a country name (level 0).
* **Frequency**: Determines the time window of data aggregation and it can be set to daily, weekly, monthly, quarterly and yearly.
* **Report Type**: Enables report filtering related to mosquito *adults*, *bites*, or *breeding sites*.

<div class="note-block">
    <p><strong> ⓘ Note: </strong> Changes in parameters affect visualizations across different views, and it may not be immediately obvious since only one view is shown at a time.</p>
</div>