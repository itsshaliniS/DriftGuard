const API_URL = '/api/v1/drift/scan';
const INFERENCE_URL = '/api/v1/predict';

let ksChartInstance = null;
let psiChartInstance = null;
let distChartInstance = null;

document.getElementById('driftForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btnParams = document.getElementById('analyzeBtn');
    const statusMsg = document.getElementById('statusMessage');
    
    btnParams.classList.add('loading');
    statusMsg.innerText = 'Analyzing data drift...';
    statusMsg.style.color = 'var(--text-secondary)';
    
    const baselineFile = document.getElementById('baselineFile').files[0];
    const productionFile = document.getElementById('productionFile').files[0];
    
    if(!baselineFile || !productionFile){
        statusMsg.style.color = 'var(--danger)';
        statusMsg.innerText = "Please upload both baseline and production CSV files.";
        btnParams.classList.remove('loading');
        return;
    }

    const formData = new FormData();
    formData.append('baseline_csv', baselineFile);
    formData.append('prod_csv', productionFile);
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        
        const json = await response.json();
        
        renderDashboard(json.data);
        renderModelPerformance(json.data.model_performance);
        
        statusMsg.style.color = 'var(--success)';
        statusMsg.innerText = "Analysis complete.";
        
    } catch (err) {
        statusMsg.style.color = 'var(--danger)';
        statusMsg.innerText = "Error analyzing files. Please check the backend server.";
        console.error(err);
    } finally {
        btnParams.classList.remove('loading');
    }
});

function renderModelPerformance(perf) {
    if (!perf || perf.error) {
        document.getElementById('accVal').innerText = "N/A";
        document.getElementById('precVal').innerText = "N/A";
        document.getElementById('recVal').innerText = "N/A";
        document.getElementById('f1Val').innerText = "N/A";
        return;
    }
    
    try {
        document.getElementById('accVal').innerText = (perf.accuracy * 100).toFixed(2) + "%";
        document.getElementById('precVal').innerText = (perf.precision * 100).toFixed(2) + "%";
        document.getElementById('recVal').innerText = (perf.recall * 100).toFixed(2) + "%";
        document.getElementById('f1Val').innerText = (perf.f1_score * 100).toFixed(2) + "%";
    } catch (perfErr) {
        console.error("Failed rendering performance strings:", perfErr);
    }
}

function renderDashboard(data) {
    try {
        const summary = data.summary;
        const details = data.feature_details;

        document.getElementById('healthScore').innerText = `${summary.dataset_health_score}%`;
        document.getElementById('driftedCount').innerText = summary.drifted_features;
        document.getElementById('totalFeatures').innerText = summary.total_features;

        const driftedCountEl = document.getElementById('driftedCount');
        if (summary.drifted_features > 0) {
            driftedCountEl.classList.add('warn-text');
            driftedCountEl.style.color = "var(--danger)";
        } else {
            driftedCountEl.classList.remove('warn-text');
            driftedCountEl.style.color = "var(--success)";
        }

        const tbody = document.getElementById('tableBody');
        tbody.innerHTML = '';
        
        const labels = [];
        const ksData = [];
        const psiData = [];
        
        let worstFeature = null;
        let worstScore = 999;
        
        details.forEach(feature => {
            labels.push(feature.feature);
            ksData.push(feature.ks_statistic || 0);
            psiData.push(feature.psi_score || 0);
            
            if (feature.feature_health < worstScore) {
                worstScore = feature.feature_health;
                worstFeature = feature;
            }
            
            const tr = document.createElement('tr');
            const isDrifted = feature.status === "Drift Detected";
            const severityClass = isDrifted ? "status-drift" : "status-ok";
            const icon = isDrifted ? "<i class='bx bx-error-circle'></i>" : "<i class='bx bx-check-circle'></i>";
            
            const ksStr = (feature.ks_statistic != null) ? feature.ks_statistic.toFixed(4) : "0.0";
            const psiStr = (feature.psi_score != null) ? feature.psi_score.toFixed(4) : "0.0";
            const jsStr = (feature.js_divergence != null) ? feature.js_divergence.toFixed(4) : "0.0";
            const severityStr = feature.psi_severity || 'Stable';
            
            tr.innerHTML = `
                <td><strong>${feature.feature}</strong></td>
                <td>${ksStr}</td>
                <td>${psiStr} <small style="color:var(--text-secondary)">(${severityStr})</small></td>
                <td>${jsStr}</td>
                <td><span class="status-badge ${severityClass}">${icon} ${feature.status}</span></td>
            `;
            
            tbody.appendChild(tr);
        });
        
        renderCharts(labels, ksData, psiData);
        if(worstFeature) renderDistributionProfiler(worstFeature);
        
    } catch(renderErr) {
        console.error("Error rendering dashboard:", renderErr);
    }
}

function renderCharts(labels, ksData, psiData) {
    try {
        const ksCtx = document.getElementById('ksChart').getContext('2d');
        const psiCtx = document.getElementById('psiChart').getContext('2d');
        
        if (ksChartInstance) ksChartInstance.destroy();
        if (psiChartInstance) psiChartInstance.destroy();
        
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Outfit', sans-serif"; 

        ksChartInstance = new Chart(ksCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'KS Statistic',
                    data: ksData,
                    backgroundColor: 'rgba(96, 165, 250, 0.6)',
                    borderColor: '#3b82f6',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } }, x: { grid: { display: false } } },
                plugins: { legend: { display: false } }
            }
        });

        const psiColors = psiData.map(val => val > 0.25 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(16, 185, 129, 0.6)');
        const psiBorders = psiData.map(val => val > 0.25 ? '#ef4444' : '#10b981');

        psiChartInstance = new Chart(psiCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        type: 'line',
                        label: 'Warning Limit (0.25)',
                        data: Array(labels.length).fill(0.25),
                        borderColor: '#ef4444',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        type: 'bar',
                        label: 'PSI Metric',
                        data: psiData,
                        backgroundColor: psiColors,
                        borderColor: psiBorders,
                        borderWidth: 1,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } }, x: { grid: { display: false } } },
                plugins: { legend: { display: true, position: 'bottom' } }
            }
        });
    } catch(chartErr) {
        console.error("Chart.js failed to paint onto the canvas:", chartErr);
    }
}

function renderDistributionProfiler(worstFeature) {
    try {
        document.getElementById('worstFeatureName').innerText = worstFeature.feature.toUpperCase();
        
        const distCtx = document.getElementById('distChart').getContext('2d');
        if (distChartInstance) distChartInstance.destroy();
        
        distChartInstance = new Chart(distCtx, {
            type: 'bar',
            data: {
                labels: worstFeature.bin_labels || ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                datasets: [
                    {
                        label: 'Baseline Batch Density',
                        data: worstFeature.base_counts || [],
                        backgroundColor: 'rgba(96, 165, 250, 0.4)',
                        borderColor: '#3b82f6',
                        borderWidth: 2,
                        borderRadius: 6
                    },
                    {
                        label: 'Drifted Production Density',
                        data: worstFeature.prod_counts || [],
                        backgroundColor: 'rgba(239, 68, 68, 0.4)',
                        borderColor: '#ef4444',
                        borderWidth: 2,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { 
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } }, 
                    x: { grid: { display: false } } 
                },
                plugins: { 
                    legend: { display: true, position: 'top' },
                    tooltip: {
                        callbacks: {
                            title: function(items) { return 'Value Bin: ' + items[0].label; }
                        }
                    }
                }
            }
        });

        const explanationEl = document.getElementById('driftExplanation');
        if (worstFeature.status === "Drift Detected") {
            const meanStr = (worstFeature.mean_shift * 100).toFixed(1);
            const varStr = (worstFeature.var_shift * 100).toFixed(1);
            let insight = "";
            
            if (worstFeature.mean_shift > 0.08) insight += `The average shifted by ${meanStr}%. `;
            if (worstFeature.var_shift > 0.1) insight += `Variance changed by ${varStr}%. `;
            if (worstFeature.psi_score > 0.2) insight += `Population stability index reached ${worstFeature.psi_score.toFixed(2)} PSI. `;
            if (insight === "") insight = `Distribution curve diverged between baseline and production batches. `;
            
            explanationEl.style.display = "block";
            explanationEl.style.borderLeftColor = "var(--danger)";
            explanationEl.innerHTML = `<i class='bx bx-brain' style='color: #8b5cf6; margin-right: 6px; font-size: 18px; vertical-align: middle;'></i><strong style="color:var(--text-primary)">Algorithmic Insight:</strong> <strong style="color:var(--danger);">${worstFeature.feature}</strong> has drifted significantly in the production data.<br><span style="color:var(--text-secondary); margin-left: 28px; display:inline-block; margin-top: 4px;">${insight} This covariate shift can degrade model predictions.</span>`;
        } else {
            explanationEl.style.display = "block";
            explanationEl.style.borderLeftColor = "var(--success)";
            explanationEl.innerHTML = `<i class='bx bx-check-shield' style='color: var(--success); margin-right: 6px; font-size: 18px; vertical-align: middle;'></i><strong style="color:var(--text-primary)">Algorithmic Insight:</strong> Distribution for <strong>${worstFeature.feature}</strong> is stable. No significant drift detected.`;
        }

    } catch(e) {
        console.error(e);
    }
}

