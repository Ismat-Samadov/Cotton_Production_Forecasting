document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent form submission

    const year = document.getElementById('year').value;

    // Generate a unique identifier to bypass cache
    const uniqueParam = new Date().getTime(); 

    // Fetch predictions from the backend with a cache-bust parameter
    fetch(`/predict_all?cache_bust=${uniqueParam}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ year: year }), // Pass the selected year to backend
    })
    .then(response => response.json())
    .then(data => {
        // Clear the old chart before displaying the new one
        const chartDiv = document.getElementById('chart');
        chartDiv.innerHTML = '';  // Clear any existing chart content
        chartDiv.innerHTML = `<img src="data:image/png;base64,${data.chart}"/>`;  // Insert new chart

        // Display predictions as a table
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '';  // Clear previous result

        const table = document.createElement('table');
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Region</th>
                    <th>Predicted Production</th>
                </tr>
            </thead>
            <tbody>
                ${data.predictions.map(pred => `
                    <tr>
                        <td>${pred.region}</td>
                        <td>${pred.production.toFixed(2)}</td>
                    </tr>
                `).join('')}
            </tbody>
        `;
        resultDiv.appendChild(table);
    })
    .catch(error => console.error('Error:', error));
});
