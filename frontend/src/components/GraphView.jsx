import { useMemo, useRef } from 'react';
import { getEdgeColor } from '../utils/colorSystem';
import GraphView2D from './GraphView2D';
import GraphView3D from './GraphView3D';

// Keep 2D responsive while allowing 3D to show the full node set.
const MAX_EDGES_2D = 12000;
const MAX_EDGES_3D = 18000;

function normalizeId(value) {
  if (value === null || value === undefined) return '';
  return String(value).trim();
}

function proxiedImageUrl(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (raw.startsWith('http://') || raw.startsWith('https://')) {
    return `/api/image-proxy?url=${encodeURIComponent(raw)}`;
  }
  return raw;
}

export default function GraphView({
  graph,
  filteredEdges,
  onNodeClick,
  selectedNodeId,
  nodeLabelById = {},
  nodeColorById = {},
  metadataById,
  viewMode,
  setViewMode,
  onZoomLevelChange,
  progressiveHint,
  simplifiedMode = false,
  thumbnailMode = false,
  attractor = 0.5,
  damping = 0.93,
  thumbnailSizePx = 100,
}) {
  const graph3dRef = useRef(null);

  // nodeData is stable when only edges change — nodes keep their positions.
  const nodeData = useMemo(() => {
    if (!graph) return [];
    return graph.nodes
      .filter((n) => normalizeId(n.id))
      .map((n) => ({
        id: normalizeId(n.id),
        label: nodeLabelById[normalizeId(n.id)] || n.label || normalizeId(n.id),
        cluster: n.cluster,
        color: nodeColorById[normalizeId(n.id)] || '#56717f',
        imageUrl: proxiedImageUrl(n.image_url || (metadataById?.get(n.instance_id || normalizeId(n.id))?.url) || ''),
      }));
  }, [graph, nodeLabelById, nodeColorById, metadataById]);

  // linkData changes when filteredEdges change — edge-only update, no node rebuild.
  const linkData = useMemo(() => {
    if (!graph) return [];
    const edgeCap = viewMode === '3d' ? MAX_EDGES_3D : MAX_EDGES_2D;
    const nodeIdSet = new Set(nodeData.map((n) => n.id));
    return [...filteredEdges]
      .sort((a, b) => b.weight - a.weight)
      .slice(0, edgeCap)
      .map((e, idx) => {
        const sourceId = normalizeId(e.source);
        const targetId = normalizeId(e.target);
        if (!sourceId || !targetId) return null;
        if (!nodeIdSet.has(sourceId) || !nodeIdSet.has(targetId)) return null;
        return {
          id: `e_${idx}`,
          source: sourceId,
          target: targetId,
          weight: e.weight,
          type: e.connection_types?.[0] || 'unknown',
          color: e.color || getEdgeColor(e, metadataById),
        };
      })
      .filter(Boolean);
  }, [graph, filteredEdges, nodeData, metadataById, viewMode]);

  // Combined for 3D (which still rebuilds on any change)
  const graphData = useMemo(() => ({ nodes: nodeData, links: linkData }), [nodeData, linkData]);

  const handleBackgroundClick = (nodeId) => {
    if (!nodeId) return;
    onNodeClick(nodeId);
  };

  return (
    <div className="graph-view-shell">
      {!simplifiedMode && (
        <div className="graph-view-toolbar">
          <div className="view-toggle" role="group" aria-label="Graph view mode">
            <button
              type="button"
              className={viewMode === '2d' ? 'view-toggle-btn active' : 'view-toggle-btn'}
              onClick={() => setViewMode('2d')}
            >
              2D
            </button>
            <button
              type="button"
              className={viewMode === '3d' ? 'view-toggle-btn active' : 'view-toggle-btn'}
              onClick={() => setViewMode('3d')}
            >
              3D
            </button>
          </div>

          {viewMode === '3d' && (
            <div className="graph-view-actions">
              <button type="button" className="view-action-btn" onClick={() => graph3dRef.current?.resetCamera()}>
                Reset Camera
              </button>
              <button
                type="button"
                className="view-action-btn"
                onClick={() => graph3dRef.current?.focusSelected()}
                disabled={!selectedNodeId}
              >
                Focus Selected
              </button>
            </div>
          )}
        </div>
      )}

      {progressiveHint && false && <p className="graph-progressive-hint subtle">{progressiveHint}</p>}

      {viewMode === '2d' ? (
        <GraphView2D
          nodeData={nodeData}
          linkData={linkData}
          onNodeClick={handleBackgroundClick}
          selectedNodeId={selectedNodeId}
          onZoomLevelChange={onZoomLevelChange}
          renderMode={thumbnailMode ? 'thumbnails' : 'dots'}
          attractor={attractor}
          damping={damping}
          thumbnailSizePx={thumbnailSizePx}
        />
      ) : (
        <GraphView3D
          ref={graph3dRef}
          graphData={graphData}
          selectedNodeId={selectedNodeId}
          onNodeClick={handleBackgroundClick}
        />
      )}
    </div>
  );
}
