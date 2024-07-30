document.getElementById('solveForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const XInput = document.getElementById('X').value;
    const yInput = document.getElementById('y').value;
    const algorithm = document.getElementById('algorithm').value;

    // Check if XInput and yInput are not empty
    if (!XInput || !yInput) {
        alert('Please provide both input and output data.');
        return;
    }

    // Parse input data
    const X = XInput.split(',').map(Number);
    const y = yInput.split(',').map(Number);

    // Validate data
    if (X.length === 0 || y.length === 0 || X.length !== y.length) {
        alert('Invalid data: Ensure X and y are comma-separated numbers and have the same length.');
        return;
    }

    const response = await fetch('/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ X, y, algorithm })
    });

    const data = await response.json();
    
    if (data.error) {
        alert('Error: ' + data.error);
        return;
    }

    // Display the graph
    const graphDiv = document.getElementById('graph');
    if (graphDiv) {
        Plotly.react(graphDiv, JSON.parse(data.graphJSON).data, JSON.parse(data.graphJSON).layout);
    } else {
        console.error('Graph div not found');
    }

    // Display the description and key points
    const descriptionElement = document.getElementById('description');
    const keyPointsList = document.getElementById('key-points');
    if (descriptionElement && keyPointsList) {
        descriptionElement.textContent = data.description;
        keyPointsList.innerHTML = '';
        data.key_points.forEach(point => {
            const li = document.createElement('li');
            li.textContent = point;
            keyPointsList.appendChild(li);
        });
    } else {
        console.error('Description or key points element not found');
    }
});
