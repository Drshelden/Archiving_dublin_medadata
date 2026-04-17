import { useEffect, useMemo, useRef, useState } from 'react';
import GraphView from './components/GraphView';
import Legend from './components/Legend';
import ChatWidget from './components/ChatWidget';
import MetadataPanel from './components/MetadataPanel';
import { createArchiveResolver } from './utils/archiveNaming';
import { getCompetitionColor } from './utils/colorSystem';

const EMPTY_GRAPH = { nodes: [], edges: [], connection_color_map: {} };

const RELATION_COLORS = [
  '#ef4444', '#0ea5e9', '#22c55e', '#eab308', '#a855f7', '#f97316',
  '#14b8a6', '#ec4899', '#6366f1', '#84cc16', '#fb7185', '#38bdf8',
];

const MAX_VISIBLE_RELATION_KEYS = 10;

const RELATION_DEFINITIONS = [
  {
    key: 'dc:subject',
    source: 'dc',
    getValues: (record) => (Array.isArray(record?.dublin_core?.['dc:subject']) ? record.dublin_core['dc:subject'] : []),
  },
  {
    key: 'archdrw:hasVisualElement',
    source: 'archdrw',
    getValues: (record) => (Array.isArray(record?.archdrw?.hasVisualElement) ? record.archdrw.hasVisualElement : []),
  },
  {
    key: 'archdrw:drawingType',
    source: 'archdrw',
    getValues: (record) => (Array.isArray(record?.archdrw?.drawingType) ? record.archdrw.drawingType : []),
  },
  {
    key: 'archdrw:buildingProgram',
    source: 'archdrw',
    getValues: (record) => (Array.isArray(record?.archdrw?.buildingProgram) ? record.archdrw.buildingProgram : []),
  },
  {
    key: 'dc:coverage',
    source: 'dc',
    getValues: (record) => [record?.dublin_core?.['dc:coverage']].filter(Boolean),
  },
  {
    key: 'dc:creator',
    source: 'dc',
    getValues: (record) => [record?.dublin_core?.['dc:creator']].filter(Boolean),
  },
  {
    key: 'archdrw:siteContext',
    source: 'archdrw',
    getValues: (record) => [record?.archdrw?.siteContext].filter(Boolean),
  },
  {
    key: 'archdrw:projection',
    source: 'archdrw',
    getValues: (record) => [record?.archdrw?.projection].filter(Boolean),
  },
  {
    key: 'year',
    source: 'node',
    getValues: (record) => (record?.year != null ? [String(record.year)] : []),
  },
  {
    key: 'dc:date',
    source: 'dc',
    getValues: (record) => {
      const v = record?.dublin_core?.['dc:date'];
      return v ? [String(v)] : [];
    },
  },
  {
    key: 'cluster',
    source: 'node',
    getValues: (record) => (record?.cluster != null ? [String(record.cluster)] : []),
  },
  {
    key: 'project_key',
    source: 'node',
    getValues: (record) => [record?.project_key].filter(Boolean),
  },
];

function normalizeToken(value) {
  return String(value || '').trim().toLowerCase();
}

function labelToken(value) {
  return String(value || '').trim();
}

function intersection(left, right) {
  const out = [];
  const rightSet = new Set(right);
  for (const item of left) {
    if (rightSet.has(item)) out.push(item);
  }
  return out;
}

function hashString(value) {
  const text = String(value || '').toLowerCase();
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

function colorForRelation(relationKey) {
  const idx = hashString(relationKey) % RELATION_COLORS.length;
  return RELATION_COLORS[idx];
}

function mergeNodeRecord(metaRecord, nodeData) {
  const panelData = nodeData && typeof nodeData === 'object' ? nodeData : {};
  if (metaRecord) {
    return {
      ...panelData,
      ...metaRecord,
      year: metaRecord.year ?? panelData.year,
      dublin_core: {
        ...(panelData.dublin_core || {}),
        ...(metaRecord.dublin_core || {}),
      },
      archdrw: {
        ...(panelData.archdrw || {}),
        ...(metaRecord.archdrw || {}),
      },
    };
  }

  return Object.keys(panelData).length ? panelData : null;
}

function inferRelationForEdge(sourceRecord, targetRecord) {
  if (!sourceRecord || !targetRecord) {
    return {
      relationKey: 'relation:unknown',
      relationLabel: 'Unknown shared relation',
      matchedRelationKeys: ['relation:unknown'],
      weightBoost: 0,
    };
  }

  let best = null;
  const matchedRelations = [];
  const seenRelationKeys = new Set();

  for (const def of RELATION_DEFINITIONS) {
    const sourceValsRaw = def.getValues(sourceRecord) || [];
    const targetValsRaw = def.getValues(targetRecord) || [];

    const sourceVals = sourceValsRaw.map(normalizeToken).filter(Boolean);
    const targetVals = targetValsRaw.map(normalizeToken).filter(Boolean);

    if (!sourceVals.length || !targetVals.length) continue;

    const overlap = intersection(sourceVals, targetVals);
    if (!overlap.length) continue;

    overlap.forEach((value) => {
      const relationKey = `${def.key}=${value}`;
      if (seenRelationKeys.has(relationKey)) return;
      seenRelationKeys.add(relationKey);
      matchedRelations.push({
        relationKey,
        relationLabel: `${def.key}: ${labelToken(value)}`,
      });
    });

    const topValue = overlap[0];
    const relationKey = `${def.key}=${topValue}`;
    const relationLabel = `${def.key}: ${labelToken(topValue)}`;

    const candidate = {
      relationKey,
      relationLabel,
      weightBoost: overlap.length,
      priority: def.source === 'dc' ? 2 : 1,
    };

    if (!best) {
      best = candidate;
      continue;
    }

    if (candidate.weightBoost > best.weightBoost) {
      best = candidate;
      continue;
    }

    if (candidate.weightBoost === best.weightBoost && candidate.priority > best.priority) {
      best = candidate;
    }
  }

  if (best) {
    return {
      ...best,
      matchedRelationKeys: matchedRelations.map((entry) => entry.relationKey),
      matchedRelationLabels: matchedRelations,
    };
  }

  return {
    relationKey: 'relation:coarse_text',
    relationLabel: 'Coarse textual relation',
    matchedRelationKeys: ['relation:coarse_text'],
    matchedRelationLabels: [
      {
        relationKey: 'relation:coarse_text',
        relationLabel: 'Coarse textual relation',
      },
    ],
    weightBoost: 0,
  };
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return response.json();
}

export default function App() {
  const [metadata, setMetadata] = useState([]);
  const [panelQueryGraph, setPanelQueryGraph] = useState(null);
  const [panelNodeToInstance, setPanelNodeToInstance] = useState({});

  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [panelNodeDataById, setPanelNodeDataById] = useState({});
  const [graphViewMode, setGraphViewMode] = useState('2d');
  const [graphZoomLevel, setGraphZoomLevel] = useState(0.28);
  const [thumbnailMode, setThumbnailMode] = useState(true);
  const [attractor, setAttractor] = useState(2500);
  const [damping, setDamping] = useState(0.6);
  const [panelReturnCount, setPanelReturnCount] = useState(20);
  const [thumbnailSizePx, setThumbnailSizePx] = useState(100);
  const [themeMode, setThemeMode] = useState('dark');
  const [relationVisibility, setRelationVisibility] = useState({});
  useEffect(() => {
    async function loadData() {
      const m = await loadJson('/data/image_metadata.json');

      let merged = m;
      try {
        const enriched = await loadJson('/data/enriched_metadata.json');
        if (Array.isArray(enriched) && enriched.length > 0) {
          const enrichedById = {};
          enriched.forEach((rec) => {
            if (rec?.instance_id) enrichedById[rec.instance_id] = rec;
          });
          merged = m.map((rec) => {
            const e = enrichedById[rec.instance_id];
            if (!e) return rec;
            return {
              ...rec,
              ...e,
              dublin_core: {
                ...(rec.dublin_core || {}),
                ...(e.dublin_core || {}),
              },
            };
          });
        }
      } catch {
        // Ignore missing enriched metadata.
      }

      setMetadata(merged);
    }

    loadData().catch((err) => {
      console.error(err);
      alert('Could not load metadata files.');
    });
  }, []);

  useEffect(() => {
    const handleKeyDown = (evt) => {
      if (evt.key === 'Tab') {
        evt.preventDefault();
        setThumbnailMode((prev) => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    document.body.classList.toggle('theme-light', themeMode === 'light');
  }, [themeMode]);

  const archiveResolver = useMemo(() => createArchiveResolver(metadata), [metadata]);

  const activeGraph = useMemo(() => {
    if (panelQueryGraph && Array.isArray(panelQueryGraph.nodes) && panelQueryGraph.nodes.length > 0) {
      return panelQueryGraph;
    }
    return EMPTY_GRAPH;
  }, [panelQueryGraph]);

  const metadataById = useMemo(() => {
    const map = new Map();
    metadata.forEach((m) => map.set(m.instance_id, m));
    return map;
  }, [metadata]);

  const drawingNameByInstanceId = useMemo(() => {
    const names = {};
    metadata.forEach((record) => {
      names[record.instance_id] = archiveResolver.getDisplayName(record);
    });
    return names;
  }, [metadata, archiveResolver]);

  const nodeColorById = useMemo(() => {
    const map = {};
    metadata.forEach((record) => {
      const key = archiveResolver.getCompetitionKey(record) || 'ungrouped';
      map[record.instance_id] = getCompetitionColor(key);
    });
    return map;
  }, [metadata, archiveResolver]);

  const selectedImage = useMemo(() => {
    const resolvedId = panelNodeToInstance[selectedNodeId] || selectedNodeId;
    return archiveResolver.resolveArchiveRecord(resolvedId) || null;
  }, [archiveResolver, selectedNodeId, panelNodeToInstance]);

  const searchDrawings = (query, terms = []) => {
    const tokenSet = new Set();
    String(query || '')
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((t) => t.length >= 2)
      .forEach((t) => tokenSet.add(t));

    (Array.isArray(terms) ? terms : [])
      .map((t) => String(t).toLowerCase())
      .flatMap((t) => t.split(/[^a-z0-9]+/))
      .filter((t) => t.length >= 2)
      .forEach((t) => tokenSet.add(t));

    const tokens = Array.from(tokenSet).slice(0, 16);
    if (!tokens.length) return [];

    const scored = [];
    for (const row of metadata) {
      const title = [
        row.resolvedDisplayTitle,
        row.canonical_board_title,
        row.board_title,
        row.title,
      ].filter(Boolean).join(' ').toLowerCase();

      const subject = Array.isArray(row?.dublin_core?.['dc:subject'])
        ? row.dublin_core['dc:subject'].join(' ').toLowerCase()
        : String(row?.dublin_core?.['dc:subject'] || '').toLowerCase();

      const keywords = Array.isArray(row?.extractedText?.keywords)
        ? row.extractedText.keywords.join(' ').toLowerCase()
        : '';

      let score = 0;
      for (const token of tokens) {
        if (title.includes(token)) score += 8;
        if (subject.includes(token)) score += 6;
        if (keywords.includes(token)) score += 7;
      }

      if (score > 0) {
        scored.push({
          score,
          instance_id: row.instance_id,
          url: row.url,
          title:
            row.resolvedDisplayTitle ||
            row.canonical_board_title ||
            row.board_title ||
            row.title ||
            row.instance_id,
        });
      }
    }

    return scored.sort((a, b) => b.score - a.score).slice(0, 120);
  };

  const openNode = (nodeId) => {
    setSelectedNodeId(nodeId);
  };

  const relationGraph = useMemo(() => {
    const nodes = Array.isArray(activeGraph?.nodes) ? activeGraph.nodes : [];
    const edges = Array.isArray(activeGraph?.edges) ? activeGraph.edges : [];

    const relationCounts = new Map();
    const relationLabelMap = {};

    const baseEdges = edges.map((edge, idx) => {
      const sourceNodeId = String(edge.source || '').trim();
      const targetNodeId = String(edge.target || '').trim();

      const sourceInstanceId = panelNodeToInstance[sourceNodeId] || sourceNodeId;
      const targetInstanceId = panelNodeToInstance[targetNodeId] || targetNodeId;

      const sourceRecord = mergeNodeRecord(
        metadataById.get(sourceInstanceId),
        panelNodeDataById[sourceNodeId],
      );
      const targetRecord = mergeNodeRecord(
        metadataById.get(targetInstanceId),
        panelNodeDataById[targetNodeId],
      );

      const relation = inferRelationForEdge(sourceRecord, targetRecord);
      const relationKey = relation.relationKey;
      const relationLabel = relation.relationLabel;

      relationCounts.set(relationKey, (relationCounts.get(relationKey) || 0) + 1);
      relationLabelMap[relationKey] = relationLabel;

      return {
        ...edge,
        id: edge.id || `edge_${idx}`,
        connection_types: [relationKey],
        relation_key: relationKey,
        color: null,
      };
    });

    const topRelationKeys = Array.from(relationCounts.entries())
      .sort((a, b) => {
        const countDiff = b[1] - a[1];
        if (countDiff !== 0) return countDiff;
        const aLabel = relationLabelMap[a[0]] || a[0];
        const bLabel = relationLabelMap[b[0]] || b[0];
        return aLabel.localeCompare(bLabel);
      })
      .slice(0, MAX_VISIBLE_RELATION_KEYS)
      .map(([key]) => key);

    const relationColorMap = {};
    topRelationKeys.forEach((key, idx) => {
      relationColorMap[key] = RELATION_COLORS[idx % RELATION_COLORS.length];
    });

    const enrichedEdges = baseEdges.map((edge) => ({
      ...edge,
      color: relationColorMap[edge.relation_key] || null,
    }));

    return {
      nodes,
      edges: enrichedEdges,
      relationCounts,
      relationColorMap,
      relationLabelMap,
      topRelationKeys,
    };
  }, [activeGraph, panelNodeToInstance, metadataById, panelNodeDataById]);

  const relationStats = useMemo(() => {
    return (relationGraph.topRelationKeys || []).map((key) => {
      const count = relationGraph.relationCounts.get(key) || 0;
      const fallbackEqIdx = key.indexOf('=');
      const fallbackLabel = fallbackEqIdx >= 0
        ? `${key.slice(0, fallbackEqIdx)}: ${labelToken(key.slice(fallbackEqIdx + 1))}`
        : key;
      return {
        key,
        count,
        color: relationGraph.relationColorMap[key] || colorForRelation(key),
        label: relationGraph.relationLabelMap[key] || fallbackLabel,
        enabled: relationVisibility[key] !== false,
      };
    });
  }, [relationGraph, relationVisibility]);

  useEffect(() => {
    const keys = relationGraph.topRelationKeys || [];
    setRelationVisibility((prev) => {
      const next = { ...prev };
      let changed = false;

      keys.forEach((key) => {
        if (!(key in next)) {
          next[key] = true;
          changed = true;
        }
      });

      Object.keys(next).forEach((key) => {
        if (!keys.includes(key)) {
          delete next[key];
          changed = true;
        }
      });

      return changed ? next : prev;
    });
  }, [relationGraph.relationColorMap]);

  const visibleRelationEdges = useMemo(() => {
    return relationGraph.edges.filter((edge) => {
      const key = edge.connection_types?.[0] || 'relation:unknown';
      if (!relationGraph.topRelationKeys?.includes(key)) return false;
      return relationVisibility[key] !== false;
    });
  }, [relationGraph.edges, relationGraph.topRelationKeys, relationVisibility]);

  const toggleRelationVisibility = (relationKey) => {
    setRelationVisibility((prev) => ({
      ...prev,
      [relationKey]: prev[relationKey] === false,
    }));
  };

  const selectedPanelData = useMemo(() => {
    if (!selectedNodeId) return null;
    return panelNodeDataById[selectedNodeId] || null;
  }, [selectedNodeId, panelNodeDataById]);

  const applyPanelGraphFromChat = (payload) => {
    const g = payload?.graph || {};
    const nodes = Array.isArray(g.nodes) ? g.nodes : [];
    const edges = Array.isArray(g.edges) ? g.edges : [];

    const idMap = {};
    const nodeDataMap = {};
    nodes.forEach((n) => {
      if (!n?.id) return;
      if (n?.instance_id) idMap[n.id] = n.instance_id;
      nodeDataMap[n.id] = n;
    });

    setPanelNodeToInstance(idMap);
    setPanelNodeDataById(nodeDataMap);
    setPanelQueryGraph({
      nodes,
      edges,
      connection_color_map: {},
    });

    const bestNode = nodes.reduce((best, current) => {
      const bestScore = Number(best?.score || Number.NEGATIVE_INFINITY);
      const currentScore = Number(current?.score || Number.NEGATIVE_INFINITY);
      return currentScore > bestScore ? current : best;
    }, null);
    setSelectedNodeId(bestNode?.id || null);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>Interconnected Drawing Archive</h1>
        <p>
          Start with chat. The graph is intentionally empty until a query runs. Press Tab to toggle node labels and thumbnail view.
        </p>
      </header>

      <div className="three-col">
        <div className="col-left">
          <div className="panel selected-image-panel">
            <h3>Selected Drawing</h3>
            {selectedImage ? (
              <img
                src={selectedImage.url}
                alt={selectedImage.title || 'Selected drawing'}
                className="selected-image-preview"
              />
            ) : (
              <p className="subtle">Select a drawing in the graph to view it here.</p>
            )}
          </div>
          <MetadataPanel
            image={selectedImage}
            drawingDisplayName={selectedImage ? drawingNameByInstanceId[selectedImage.instance_id] : null}
            archiveSecondaryLine={selectedImage ? archiveResolver.getSecondaryLine(selectedImage) : null}
            onOpenDrawing={openNode}
          />
        </div>

        <main className="col-center graph-column">
          <GraphView
            graph={{ nodes: relationGraph.nodes, edges: visibleRelationEdges }}
            filteredEdges={visibleRelationEdges}
            onNodeClick={openNode}
            selectedNodeId={selectedNodeId}
            nodeLabelById={drawingNameByInstanceId}
            nodeColorById={nodeColorById}
            metadataById={metadataById}
            viewMode={graphViewMode}
            setViewMode={setGraphViewMode}
            onZoomLevelChange={setGraphZoomLevel}
            progressiveHint={relationGraph.nodes.length === 0 ? 'No results yet. Submit a chat query to populate the graph.' : `Showing ${relationGraph.nodes.length} retrieved panels / ${visibleRelationEdges.length} visible relations at zoom ${graphZoomLevel.toFixed(2)}.`}
            simplifiedMode
            thumbnailMode={thumbnailMode}
            attractor={attractor}
            damping={damping}
            thumbnailSizePx={thumbnailSizePx}
          />
        </main>

        <div className="col-right">
          <Legend
            title="Panel Relations Key"
            relationStats={relationStats}
            onToggleRelation={toggleRelationVisibility}
          />
          <ChatWidget
            selectedImage={selectedImage}
            archiveSecondaryLine={selectedImage ? archiveResolver.getSecondaryLine(selectedImage) : null}
            totalDrawings={metadata.length}
            onOpenDrawing={openNode}
            searchDrawings={searchDrawings}
            onApplyPanelGraph={applyPanelGraphFromChat}
            minPanels={5}
            maxPanels={50}
            panelReturnCount={panelReturnCount}
            setPanelReturnCount={setPanelReturnCount}
            attractor={attractor}
            setAttractor={setAttractor}
            damping={damping}
            setDamping={setDamping}
            thumbnailSizePx={thumbnailSizePx}
            setThumbnailSizePx={setThumbnailSizePx}
            themeMode={themeMode}
            setThemeMode={setThemeMode}
          />
        </div>
      </div>
    </div>
  );
}
