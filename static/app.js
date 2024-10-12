document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const year = document.getElementById('year').value;

    try {
        const response = await fetch(`/predict?year=${year}`);
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Check if prediction is a number
        if (data.prediction && !isNaN(data.prediction)) {
            document.getElementById('result').innerText = data.prediction;  // Output the prediction
        } else {
            throw new Error('Prediction is not a number');
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'Error: ' + error.message;
    }
});
