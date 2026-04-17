import { useEffect, useMemo, useState } from 'react';
import GraphView from './components/GraphView';
import Legend from './components/Legend';
import ChatWidget from './components/ChatWidget';
import { createArchiveResolver } from './utils/archiveNaming';
import { getCompetitionColor } from './utils/colorSystem';

const EMPTY_GRAPH = { nodes: [], edges: [], connection_color_map: {} };

const RELATION_COLORS = [
  '#ef4444', '#0ea5e9', '#22c55e', '#eab308', '#a855f7', '#f97316',
  '#14b8a6', '#ec4899', '#6366f1', '#84cc16', '#fb7185', '#38bdf8',
];

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
    const relationColorMap = {};
    const relationLabelMap = {};

    const enrichedEdges = edges.map((edge, idx) => {
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
      const matchedRelationKeys = relation.matchedRelationKeys || [relationKey];
      const matchedRelationLabels = relation.matchedRelationLabels || [{ relationKey, relationLabel }];

      relationCounts.set(relationKey, (relationCounts.get(relationKey) || 0) + 1);
      matchedRelationKeys.forEach((key) => {
        if (!relationColorMap[key]) {
          relationColorMap[key] = colorForRelation(key);
        }
      });
      matchedRelationLabels.forEach(({ relationKey: key, relationLabel: label }) => {
        relationLabelMap[key] = label;
      });

      return {
        ...edge,
        id: edge.id || `edge_${idx}`,
        connection_types: [relationKey],
        matched_relation_keys: matchedRelationKeys,
        color: relationColorMap[relationKey],
      };
    });

    return {
      nodes,
      edges: enrichedEdges,
      relationCounts,
      relationColorMap,
      relationLabelMap,
    };
  }, [activeGraph, panelNodeToInstance, metadataById, panelNodeDataById]);

  const relationStats = useMemo(() => {
    const selected = String(selectedNodeId || '').trim();
    const relevantEdges = selected
      ? relationGraph.edges.filter((edge) => {
        const sourceId = String(edge.source || '').trim();
        const targetId = String(edge.target || '').trim();
        return sourceId === selected || targetId === selected;
      })
      : relationGraph.edges;

    const counts = new Map();

    relevantEdges.forEach((edge) => {
      const keys = Array.isArray(edge.matched_relation_keys) && edge.matched_relation_keys.length
        ? edge.matched_relation_keys
        : [edge.connection_types?.[0] || 'relation:unknown'];

      keys.forEach((key) => {
        counts.set(key, (counts.get(key) || 0) + 1);
      });
    });

    return Array.from(counts.entries())
      .map(([key, count]) => {
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
      })
      .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
      .slice(0, 20);
  }, [relationGraph, selectedNodeId, relationVisibility]);

  useEffect(() => {
    const keys = Object.keys(relationGraph.relationColorMap || {});
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
      const keys = Array.isArray(edge.matched_relation_keys) && edge.matched_relation_keys.length
        ? edge.matched_relation_keys
        : [edge.connection_types?.[0] || 'relation:unknown'];
      return keys.every((key) => relationVisibility[key] !== false);
    });
  }, [relationGraph.edges, relationVisibility]);

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
      <div className="app-left">
        <header className="topbar">
          <h1>Interconnected Drawing Archive</h1>
          <p>
            Start with chat. The graph is intentionally empty until a query runs. Press Tab to toggle node labels and thumbnail view.
          </p>
        </header>

        <section className="workspace workspace-graph-only">
          <main className="graph-column">
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
        </section>
      </div>

      <aside className="app-right">
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

        <Legend
          title="Panel Relations Key"
          relationStats={relationStats}
          onToggleRelation={toggleRelationVisibility}
        />

        {(selectedPanelData || selectedImage) && (
          <div className="selected-item-data">
            <h4>Selected Item Data</h4>
            {selectedPanelData && (
              <>
                <p className="selected-item-data__section-label">Panel</p>
                <pre>{JSON.stringify(selectedPanelData, null, 2)}</pre>
              </>
            )}
            {selectedImage && (
              <>
                <p className="selected-item-data__section-label">Image Metadata</p>
                <pre>{JSON.stringify(selectedImage, null, 2)}</pre>
              </>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}
