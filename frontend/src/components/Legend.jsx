export default function Legend({ relationStats = [], title = 'Link Meanings', onToggleRelation }) {

  return (
    <div className="legend">
      <h3>{title}</h3>
      <p className="subtle legend-intro">Most common Dublin Core / archdrw key–value pairs across the displayed images.</p>

      {relationStats.length === 0 && (
        <p className="subtle">No relation statistics yet. Submit a query or select a node.</p>
      )}

      {relationStats.map((item) => (
        <button
          type="button"
          key={item.key}
          className={item.enabled ? 'legend-item' : 'legend-item legend-item--off'}
          onClick={() => onToggleRelation && onToggleRelation(item.key)}
          title={item.enabled ? 'Turn relation off' : 'Turn relation on'}
        >
          <span className="legend-swatch" style={{ backgroundColor: item.color }} />
          <span className="legend-label">{item.label}</span>
          <span className="legend-count">{item.count}</span>
        </button>
      ))}
    </div>
  );
}
