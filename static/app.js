document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const year = document.getElementById('year').value;

    try {
        const response = await fetch(`/predict?year=${year}`);
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Check if predictions are an array
        if (data.predictions && Array.isArray(data.predictions)) {
            // Beautify the prediction output
            const resultContainer = document.getElementById('result');
            resultContainer.innerHTML = `
                <h3>🌾 Predicted Cotton Production for the Next 5 Years 🌾</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background-color: #f2f2f2;">
                            <th style="padding: 8px; border: 1px solid #ddd;">Year</th>
                            <th style="padding: 8px; border: 1px solid #ddd;">Production (tons)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.predictions.map((prediction, index) => `
                            <tr>
                                <td style="padding: 8px; border: 1px solid #ddd;">${2024 + index}</td>
                                <td style="padding: 8px; border: 1px solid #ddd;">${prediction.toFixed(2)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        } else {
            throw new Error('Predictions are not in the expected format');
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'Error: ' + error.message;
    }
});
