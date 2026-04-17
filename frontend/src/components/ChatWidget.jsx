import { useMemo, useState } from 'react';
import { fetchPanelQueryGraph } from '../utils/backendApi';



export default function ChatWidget({
  selectedImage,
  archiveSecondaryLine,
  totalDrawings = 0,
  onOpenDrawing,
  searchDrawings,
  onApplyPanelGraph,
  minPanels = 5,
  maxPanels = 50,
  panelReturnCount = 20,
  setPanelReturnCount,
  attractor = 2500,
  setAttractor,
  damping = 0.6,
  setDamping,
  thumbnailSizePx = 100,
  setThumbnailSizePx,
  themeMode = 'dark',
  setThemeMode,
}) {
  const [isLoading, setIsLoading] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Ask for panels and I will load a graph of the most relevant results. Press Tab to switch to thumbnail view.',
    },
  ]);
  const [error, setError] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [lastSearchLabel, setLastSearchLabel] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);

  const selectedContext = useMemo(() => {
    if (!selectedImage) return null;
    return {
      title:
        selectedImage.resolvedDisplayTitle ||
        selectedImage.canonical_board_title ||
        selectedImage.board_title ||
        selectedImage.title ||
        'Untitled drawing',
      instanceId: selectedImage.instance_id || '',
      secondaryLine: archiveSecondaryLine || '',
    };
  }, [selectedImage, archiveSecondaryLine]);

  const sendMessage = async (forcedText = null) => {
    const text = (forcedText ?? input).trim();
    if (!text || isLoading) return;

    setError('');
    setInput('');

    const nextMessages = [...messages, { role: 'user', content: text }];
    setMessages(nextMessages);
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: nextMessages,
          selectedContext,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.error || 'Chat request failed');
      }

      setMessages((prev) => [...prev, { role: 'assistant', content: payload.answer }]);

      const search = payload?.search || { enabled: false, query: '', terms: [] };
      const queryText = search.query || text;

      if (typeof searchDrawings === 'function') {
        const matches = searchDrawings(queryText, search.terms || []);
        setSearchResults(matches);
        setLastSearchLabel(queryText);
      }

      if (typeof onApplyPanelGraph === 'function') {
        const panelGraphPayload = await fetchPanelQueryGraph({
          query: queryText,
          topK: panelReturnCount,
          maxEdges: Math.max(panelReturnCount * 8, 80),
        });

        if (panelGraphPayload?.ok) {
          onApplyPanelGraph(panelGraphPayload);
        }
      }
    } catch (err) {
      setError(err?.message || 'Unable to send message');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-widget chat-widget--always-open">
      <div className="chat-widget-panel">
        <div className="chat-widget-header">
          <div className="chat-header-row">
            <h3>Panel Retrieval</h3>
            <button
              type="button"
              className="chat-settings-btn"
              onClick={() => setSettingsOpen((v) => !v)}
            >
              {settingsOpen ? 'Close Settings' : 'Settings'}
            </button>
          </div>
          <p className="subtle">Dataset loaded: {totalDrawings} drawings</p>

          {settingsOpen && (
            <div className="chat-settings-panel">
              <label>
                Retrieved Panels: {panelReturnCount}
                <input
                  type="range"
                  min={minPanels}
                  max={maxPanels}
                  step={1}
                  value={panelReturnCount}
                  onChange={(e) => setPanelReturnCount && setPanelReturnCount(Math.max(minPanels, Math.min(maxPanels, Number(e.target.value) || 20)))}
                />
              </label>

              <label>
                Repel/Attract: {Number(attractor).toFixed(1)}x
                <input
                  type="range"
                  min={0.1}
                  max={9999}
                  step={0.1}
                  value={attractor}
                  onChange={(e) => setAttractor && setAttractor(Number(e.target.value))}
                />
              </label>

              <label>
                Damping: {Number(damping).toFixed(2)}
                <input
                  type="range"
                  min={0}
                  max={5}
                  step={0.01}
                  value={damping}
                  onChange={(e) => setDamping && setDamping(Number(e.target.value))}
                />
              </label>

              <label>
                Thumbnail Size: {Math.round(thumbnailSizePx)}px
                <input
                  type="range"
                  min={50}
                  max={200}
                  step={1}
                  value={thumbnailSizePx}
                  onChange={(e) => setThumbnailSizePx && setThumbnailSizePx(Number(e.target.value))}
                />
              </label>

              <label>
                Theme: {themeMode === 'light' ? 'Light' : 'Dark'}
                <button
                  type="button"
                  className="chat-theme-toggle-btn"
                  onClick={() => setThemeMode && setThemeMode(themeMode === 'light' ? 'dark' : 'light')}
                >
                  Switch to {themeMode === 'light' ? 'Dark' : 'Light'}
                </button>
              </label>
            </div>
          )}

        </div>

        <div className="chat-widget-messages">
          {messages.map((m, idx) => (
            <div key={`${m.role}-${idx}`} className={`chat-message chat-message--${m.role}`}>
              <span className="chat-role">{m.role === 'assistant' ? 'Assistant' : 'You'}</span>
              <p>{m.content}</p>
            </div>
          ))}
          {isLoading && <p className="subtle">Loading results...</p>}

          {searchResults.length > 0 && (
            <div className="chat-results">
              <p className="chat-results-title">
                Matched drawings ({searchResults.length})
                {lastSearchLabel ? ` for "${lastSearchLabel}"` : ''}
              </p>
              <div className="chat-results-grid">
                {searchResults.slice(0, 12).map((r) => (
                  <button
                    type="button"
                    key={r.instance_id}
                    className="chat-result-card"
                    onClick={() => onOpenDrawing && onOpenDrawing(r.instance_id)}
                    title="Open drawing"
                  >
                    <img src={r.url} alt={r.title} loading="lazy" />
                    <span className="chat-result-title">{r.title}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {error && <p className="chat-error">{error}</p>}

        <div className="chat-widget-input-row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Describe what panels to retrieve… (Enter to send, Shift+Enter for newline)"
            rows={3}
          />
        </div>
      </div>
    </div>
  );
}
