function updateInterventionROI() {
    const country = document.getElementById('country-select').value;
    const nutritionVal = parseFloat(document.getElementById('nutrition-slider').value) || 0;
    const sanitationVal = parseFloat(document.getElementById('sanitation-slider').value) || 0;
    const literacyVal = parseFloat(document.getElementById('literacy-slider').value) || 0;
    const gdpVal = parseFloat(document.getElementById('gdp-slider').value) || 0;

    let burdenMultiplier = 1.0;
    if (['Afghanistan', 'Ethiopia', 'Congo, Dem. Rep.', 'Pakistan'].includes(country)) {
        burdenMultiplier = 0.82; // Higher intervention cost-efficiency in high-burden nations
    } else if (country === 'China') {
        burdenMultiplier = 1.15;
    }

    const dynNutrition = (124.50 * burdenMultiplier * (1 - nutritionVal * 0.005)).toFixed(2);
    const dynSanitation = (210.00 * burdenMultiplier * (1 - sanitationVal * 0.005)).toFixed(2);
    const dynLiteracy = (850.25 * burdenMultiplier * (1 - literacyVal * 0.005)).toFixed(2);
    const dynGdp = (3200.00 * burdenMultiplier * (1 - gdpVal * 0.005)).toFixed(2);

    document.getElementById('roi-nutrition-cost').textContent = `$${Number(dynNutrition).toLocaleString()}`;
    document.getElementById('roi-sanitation-cost').textContent = `$${Number(dynSanitation).toLocaleString()}`;
    document.getElementById('roi-literacy-cost').textContent = `$${Number(dynLiteracy).toLocaleString()}`;
    document.getElementById('roi-gdp-cost').textContent = `$${Number(dynGdp).toLocaleString()}`;
}

document.querySelectorAll('input[type="range"], select').forEach(el => {
    el.addEventListener('input', updateInterventionROI);
    el.addEventListener('change', updateInterventionROI);
});