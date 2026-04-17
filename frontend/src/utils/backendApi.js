export async function explainMatch(payload) {
  const response = await fetch('/api/explain-match', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error('Backend explain-match request failed');
  }

  return response.json();
}

export async function extractBoardTitle(payload) {
  const response = await fetch('/api/extract-board-title', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error('Backend extract-board-title request failed');
  return response.json();
}

export async function extractImageMetadata(payload) {
  const response = await fetch('/api/extract-image-metadata', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error('Backend extract-image-metadata request failed');
  return response.json();
}

export async function fetchSimilarDrawings({ instanceId, drawingRef, topK = 12 }) {
  const params = new URLSearchParams();
  if (instanceId) params.set('instance_id', instanceId);
  if (drawingRef) params.set('drawing_ref', drawingRef);
  params.set('top_k', String(topK));

  const response = await fetch(`/api/similar-drawings?${params.toString()}`);
  if (!response.ok) throw new Error('Backend similar-drawings request failed');
  return response.json();
}

export async function fetchSimilarPanels({ panelRef, topK = 16 }) {
  const params = new URLSearchParams();
  params.set('panel_ref', panelRef);
  params.set('top_k', String(topK));

  const response = await fetch(`/api/similar-panels?${params.toString()}`);
  if (!response.ok) throw new Error('Backend similar-panels request failed');
  return response.json();
}

export async function fetchPanelQueryGraph({ query, topK = 24, maxEdges = 240, minScore = 0.05 }) {
  const response = await fetch('/api/panel-query-graph', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      top_k: topK,
      max_edges: maxEdges,
      min_score: minScore,
    }),
  });

  if (!response.ok) throw new Error('Backend panel-query-graph request failed');
  return response.json();
}
