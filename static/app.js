document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const year = document.getElementById('year').value;
    const region = document.getElementById('region').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ year: year, region: region }),
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `Predicted Cotton Production: ${data.predicted_production.toFixed(2)}`;
    })
    .catch(error => console.error('Error:', error));
});
