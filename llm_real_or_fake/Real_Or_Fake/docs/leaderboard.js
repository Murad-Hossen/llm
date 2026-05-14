async function loadCSV() {
  const response = await fetch('leaderboard.csv');
  if (!response.ok) {
    throw new Error('Failed to load leaderboard.csv');
  }
  const text = await response.text();
  return parseCSV(text);
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(',');
  return lines.map(line => {
    const values = line.split(',');
    const row = {};
    headers.forEach((h, i) => row[h] = values[i]);
    return row;
  });
}

function renderTable(rows) {
  const tbody = document.querySelector('#leaderboard tbody');
  tbody.innerHTML = '';
  rows.forEach(row => {
    const submitter = row.submitter || '';
    const submitterLink = submitter
      ? `<a href="https://github.com/${submitter}" target="_blank" rel="noopener noreferrer">${submitter}</a>`
      : '';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.rank}</td>
      <td>${row.team}</td>
      <td>${row.run_id}</td>
      <td>${row.model}</td>
      <td>${row.model_type}</td>
      <td>${Number(row.f1_score).toFixed(4)}</td>
      <td>${Number(row.accuracy).toFixed(4)}</td>
   
      <td>${row.submission_date}</td>
      
    `;
    tbody.appendChild(tr);
  });
}

function applyFilters(rows) {
  const q = document.getElementById('search').value.toLowerCase();
  const type = document.getElementById('typeFilter').value;
  const sortBy = document.getElementById('sortBy').value;

  let filtered = rows.filter(r => {
    const hay = `${r.team} ${r.run_id} ${r.model} ${r.model_type} ${r.submitter}`.toLowerCase();
    const matchesText = hay.includes(q);
    const matchesType = type === 'all' || (r.model_type || 'unknown') === type;
    return matchesText && matchesType;
  });

  filtered.sort((a, b) => {
    if (sortBy === 'submission_date') {
      return (b.submission_date || '').localeCompare(a.submission_date || '');
    }
    return Number(b[sortBy]) - Number(a[sortBy]);
  });

  renderTable(filtered);
}

(async () => {
  try {
    const rows = await loadCSV();
    renderTable(rows);

    document.getElementById('search').addEventListener('input', () => applyFilters(rows));
    document.getElementById('typeFilter').addEventListener('change', () => applyFilters(rows));
    document.getElementById('sortBy').addEventListener('change', () => applyFilters(rows));
  } catch (e) {
    console.error(e);
  }
})();
