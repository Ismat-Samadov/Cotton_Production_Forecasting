// Fetch and populate the region dropdown
async function fetchRegions() {
    const response = await fetch('/regions');
    const data = await response.json();

    const regionSelect = document.getElementById('regionSelect');
    data.regions.forEach(region => {
        const option = document.createElement('option');
        option.value = region;
        option.text = region;
        regionSelect.appendChild(option);
    });
}

// Fetch total production forecast
async function fetchTotalProduction() {
    const response = await fetch('/predict/whole');
    const data = await response.json();
    const graphJSON = data.graph;

    // Plot the total production graph
    Plotly.newPlot('totalProductionGraph', JSON.parse(graphJSON).data, JSON.parse(graphJSON).layout);
}

// Fetch economic region forecast
async function fetchEconomicRegion() {
    const response = await fetch('/predict/economic-region');
    const data = await response.json();
    const graphJSON = data.graph;

    // Plot the economic region graph
    Plotly.newPlot('economicRegionGraph', JSON.parse(graphJSON).data, JSON.parse(graphJSON).layout);
}

// Fetch production forecast for selected region
async function fetchRegionForecast(region) {
    const response = await fetch(`/predict/region?region=${region}`);
    const data = await response.json();
    
    if (data.graph) {
        const graphJSON = data.graph;
        Plotly.newPlot('regionGraph', JSON.parse(graphJSON).data, JSON.parse(graphJSON).layout);
    } else {
        document.getElementById('regionGraph').innerText = 'No data available for the selected region.';
    }
}

// Initialize the page by fetching regions and charts
document.addEventListener('DOMContentLoaded', async () => {
    await fetchRegions();
    await fetchTotalProduction();
    await fetchEconomicRegion();

    // Handle region change event
    const regionSelect = document.getElementById('regionSelect');
    regionSelect.addEventListener('change', async function() {
        const selectedRegion = regionSelect.value;
        await fetchRegionForecast(selectedRegion);
    });

    // Fetch default region forecast (first region in the list)
    const firstRegion = regionSelect.value;
    if (firstRegion) {
        await fetchRegionForecast(firstRegion);
    }
});
