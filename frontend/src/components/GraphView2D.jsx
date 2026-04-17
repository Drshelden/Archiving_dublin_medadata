import { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';

const DOT_SCREEN_PX = 30;
const THUMB_SCREEN_PX_DEFAULT = 155;
const LABEL_FONT_PX = 12;
const LABEL_MAX_WIDTH_PX = 150;
const LABEL_MARGIN_Y_PX = 8;
const SELECTED_BORDER_PX = 2;
const VIEW_BOUNDS_PADDING_PX = 26;

function clampRatio(value) {
  if (Number.isNaN(value)) return 1;
  return Math.min(9999, Math.max(0.1, value));
}

function clampDamping(value) {
  if (Number.isNaN(value)) return 0.6;
  return Math.min(5, Math.max(0, value));
}

function clampThumbSize(value) {
  if (Number.isNaN(value)) return THUMB_SCREEN_PX_DEFAULT;
  return Math.min(200, Math.max(50, value));
}

function lerp(min, max, t) {
  return min + (max - min) * t;
}

function syncScreenSpaceSizing(cy, renderMode, thumbnailSizePx) {
  const zoom = Math.max(0.001, cy.zoom());
  const thumbSize = clampThumbSize(thumbnailSizePx);
  const dotSize = (renderMode === 'thumbnails' ? thumbSize : DOT_SCREEN_PX) / zoom;
  const fontSize = LABEL_FONT_PX / zoom;
  const maxWidth = LABEL_MAX_WIDTH_PX / zoom;
  const marginY = LABEL_MARGIN_Y_PX / zoom;
  const selectedBorder = SELECTED_BORDER_PX / zoom;

  const textOpacity = renderMode === 'thumbnails' ? 0 : 1;

  cy.style()
    .selector('node')
    .style({
      width: dotSize,
      height: dotSize,
      'font-size': fontSize,
      'text-max-width': maxWidth,
      'text-margin-y': marginY,
      'text-opacity': textOpacity,
    })
    .selector('node.hovered')
    .style({
      width: dotSize,
      height: dotSize,
      'font-size': fontSize,
      'text-max-width': maxWidth,
      'text-margin-y': marginY,
      'text-opacity': textOpacity,
    })
    .selector('node:selected')
    .style({
      width: dotSize,
      height: dotSize,
      'font-size': fontSize,
      'text-max-width': maxWidth,
      'text-margin-y': marginY,
      'border-width': selectedBorder,
      'text-opacity': textOpacity,
    })
    .update();
}

function constrainNodeToViewport(node, cy) {
  const extent = cy.extent();
  const zoom = Math.max(0.001, cy.zoom());
  const pad = VIEW_BOUNDS_PADDING_PX / zoom;

  const minX = extent.x1 + pad;
  const maxX = extent.x2 - pad;
  const minY = extent.y1 + pad;
  const maxY = extent.y2 - pad;

  const pos = node.position();
  const x = Math.min(maxX, Math.max(minX, pos.x));
  const y = Math.min(maxY, Math.max(minY, pos.y));

  const hitX = x !== pos.x;
  const hitY = y !== pos.y;
  if (hitX || hitY) {
    node.position({ x, y });
  }

  return { hitX, hitY };
}

function runForceStep(cy, attractor, dampingValue, selectedNodeId, velocityById) {
  const ratio = clampRatio(attractor);
  const nodes = cy.nodes();
  const edges = cy.edges();
  const n = nodes.length;
  if (!n) return;

  const springK = 0.04;
  const repulsionK = 1400 * ratio;
  const preferredLength = lerp(130, 260, Math.min(1, ratio / 9999));
  const dampingStrength = clampDamping(dampingValue);
  const damping = 1 / (1 + dampingStrength * 0.6);
  const maxSpeed = 9;
  const selectedCenterK = 0.26;

  const forceById = new Map();
  const nodeById = new Map();

  for (let i = 0; i < n; i += 1) {
    const node = nodes[i];
    const id = node.id();
    nodeById.set(id, node);
    forceById.set(id, { x: 0, y: 0 });
    if (!velocityById.has(id)) velocityById.set(id, { x: 0, y: 0 });
  }

  for (let i = 0; i < n; i += 1) {
    const a = nodes[i];
    const aId = a.id();
    const pa = a.position();
    for (let j = i + 1; j < n; j += 1) {
      const b = nodes[j];
      const bId = b.id();
      const pb = b.position();

      const dx = pb.x - pa.x;
      const dy = pb.y - pa.y;
      const d2 = Math.max(36, (dx * dx) + (dy * dy));
      const d = Math.sqrt(d2);
      const inv = 1 / d;

      const f = repulsionK / d2;
      const fx = f * dx * inv;
      const fy = f * dy * inv;

      const fa = forceById.get(aId);
      const fb = forceById.get(bId);
      fa.x -= fx;
      fa.y -= fy;
      fb.x += fx;
      fb.y += fy;
    }
  }

  for (let i = 0; i < edges.length; i += 1) {
    const edge = edges[i];
    const source = edge.source();
    const target = edge.target();

    const ps = source.position();
    const pt = target.position();

    const dx = pt.x - ps.x;
    const dy = pt.y - ps.y;
    const d = Math.max(1, Math.sqrt((dx * dx) + (dy * dy)));
    const ux = dx / d;
    const uy = dy / d;

    const weight = Number(edge.data('weight') || 0.5);
    const naturalLen = preferredLength * (1.2 - Math.min(1, weight));
    const stretch = d - naturalLen;
    const pull = springK * stretch;

    const fs = forceById.get(source.id());
    const ft = forceById.get(target.id());

    fs.x += pull * ux;
    fs.y += pull * uy;
    ft.x -= pull * ux;
    ft.y -= pull * uy;
  }

  const extent = cy.extent();
  const center = { x: (extent.x1 + extent.x2) / 2, y: (extent.y1 + extent.y2) / 2 };

  if (selectedNodeId) {
    const selectedNode = nodeById.get(selectedNodeId);
    if (selectedNode && !selectedNode.grabbed()) {
      const p = selectedNode.position();
      const f = forceById.get(selectedNodeId);
      f.x += (center.x - p.x) * selectedCenterK;
      f.y += (center.y - p.y) * selectedCenterK;
    }
  }

  for (let i = 0; i < n; i += 1) {
    const node = nodes[i];
    const id = node.id();

    if (node.grabbed()) {
      velocityById.set(id, { x: 0, y: 0 });
      continue;
    }

    const v = velocityById.get(id) || { x: 0, y: 0 };
    const f = forceById.get(id) || { x: 0, y: 0 };

    let vx = (v.x + f.x) * damping;
    let vy = (v.y + f.y) * damping;

    if (Math.abs(vx) < 0.015) vx = 0;
    if (Math.abs(vy) < 0.015) vy = 0;

    const speed = Math.sqrt((vx * vx) + (vy * vy));
    if (speed > maxSpeed) {
      const s = maxSpeed / speed;
      vx *= s;
      vy *= s;
    }

    const p = node.position();
    node.position({ x: p.x + vx, y: p.y + vy });

    const hit = constrainNodeToViewport(node, cy);
    if (hit.hitX) vx *= -0.45;
    if (hit.hitY) vy *= -0.45;

    velocityById.set(id, { x: vx, y: vy });
  }
}

export default function GraphView2D({ nodeData, linkData, onNodeClick, selectedNodeId, onZoomLevelChange, renderMode = 'dots', attractor = 1, damping = 0.6, thumbnailSizePx = THUMB_SCREEN_PX_DEFAULT }) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const onClickRef = useRef(onNodeClick);
  const onZoomRef = useRef(onZoomLevelChange);
  const rafRef = useRef(null);
  const simulationRef = useRef(null);
  const selectedRef = useRef(selectedNodeId);
  const attractorRef = useRef(attractor);
  const dampingRef = useRef(damping);
  const thumbSizeRef = useRef(thumbnailSizePx);
  const velocityByIdRef = useRef(new Map());
  const linkDataRef = useRef(linkData);

  useEffect(() => {
    onZoomRef.current = onZoomLevelChange;
  }, [onZoomLevelChange]);

  useEffect(() => {
    onClickRef.current = onNodeClick;
  }, [onNodeClick]);

  useEffect(() => {
    selectedRef.current = selectedNodeId;
  }, [selectedNodeId]);

  useEffect(() => {
    attractorRef.current = attractor;
  }, [attractor]);

  useEffect(() => {
    dampingRef.current = damping;
  }, [damping]);

  useEffect(() => {
    thumbSizeRef.current = thumbnailSizePx;
  }, [thumbnailSizePx]);

  // NODE REBUILD — only runs when nodes or renderMode change; preserves positions of surviving nodes.
  useEffect(() => {
    if (!containerRef.current || !nodeData) return;

    // Snapshot positions before destroying the old instance.
    const savedPositions = new Map();
    const existingCy = cyRef.current;
    if (existingCy) {
      existingCy.nodes().forEach((n) => {
        savedPositions.set(n.id(), { ...n.position() });
      });
      existingCy.destroy();
      cyRef.current = null;
    }

    if (simulationRef.current) {
      cancelAnimationFrame(simulationRef.current);
      simulationRef.current = null;
    }

    // Preserve velocities only for surviving nodes.
    const oldVelocities = velocityByIdRef.current;
    const newVelocities = new Map();
    nodeData.forEach((n) => {
      if (oldVelocities.has(n.id)) newVelocities.set(n.id, oldVelocities.get(n.id));
    });
    velocityByIdRef.current = newVelocities;

    const cyNodes = nodeData.map((n) => {
      const saved = savedPositions.get(n.id);
      return {
        data: { id: n.id, label: n.label, cluster: n.cluster, color: n.color, image: n.imageUrl || '' },
        ...(saved ? { position: saved } : {}),
      };
    });

    const currentLinks = linkDataRef.current || [];
    const nodeIdSet = new Set(nodeData.map((n) => n.id));
    const cyEdges = currentLinks
      .map((e) => {
        const sourceId = typeof e.source === 'object' ? e.source?.id : e.source;
        const targetId = typeof e.target === 'object' ? e.target?.id : e.target;
        if (!sourceId || !targetId || !nodeIdSet.has(sourceId) || !nodeIdSet.has(targetId)) return null;
        return {
          data: { id: e.id, source: sourceId, target: targetId, weight: e.weight, color: e.color },
        };
      })
      .filter(Boolean);

    const cy = cytoscape({
      container: containerRef.current,
      elements: [...cyNodes, ...cyEdges],
      wheelSensitivity: 0.2,
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: false,
      autoungrabify: false,
      autounselectify: false,
      minZoom: 0.03,
      maxZoom: 7,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'background-fit': 'cover',
            'background-image': renderMode === 'thumbnails' ? 'data(image)' : 'none',
            shape: renderMode === 'thumbnails' ? 'round-rectangle' : 'ellipse',
            label: 'data(label)',
            'font-family': 'Space Grotesk, sans-serif',
            'font-size': LABEL_FONT_PX,
            color: '#e2e8f0',
            'text-background-color': 'rgba(3,11,24,0.72)',
            'text-background-opacity': 1,
            'text-background-padding': '3px',
            'text-wrap': 'ellipsis',
            'text-max-width': LABEL_MAX_WIDTH_PX,
            'text-valign': 'bottom',
            'text-margin-y': LABEL_MARGIN_Y_PX,
          },
        },
        {
          selector: 'node.hovered',
          style: {
            color: '#ffffff',
            'text-background-color': 'rgba(3,11,24,0.88)',
            'text-background-opacity': 1,
            'z-index': 5,
          },
        },
        {
          selector: 'node:selected',
          style: {
            color: '#ffffff',
            'text-background-color': 'rgba(3,11,24,0.9)',
            'text-background-opacity': 1,
            'text-background-padding': '3px',
            'background-color': renderMode === 'thumbnails' ? '#ffffff' : '#fde047',
            'border-width': SELECTED_BORDER_PX,
            'border-color': '#fff4b8',
            'z-index': 10,
          },
        },
        {
          selector: 'edge',
          style: {
            width: 'mapData(weight, 0, 1, 0.4, 3)',
            'line-color': 'data(color)',
            opacity: 0.48,
            'curve-style': 'haystack',
          },
        },
      ],
      layout: savedPositions.size === 0
        ? { name: 'circle', fit: true, padding: 40 }
        : { name: 'preset', fit: false },
    });

    cy.on('mouseover', 'node', (evt) => evt.target.addClass('hovered'));
    cy.on('mouseout', 'node', (evt) => evt.target.removeClass('hovered'));

    const notifyZoom = () => {
      if (!onZoomRef.current) return;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => {
        onZoomRef.current(cy.zoom());
      });
    };

    cy.on('zoom', () => {
      syncScreenSpaceSizing(cy, renderMode, thumbSizeRef.current);
      notifyZoom();
    });

    cy.on('tap', 'node', (evt) => {
      onClickRef.current(evt.target.id());
      cy.nodes().unselect();
      evt.target.select();
    });

    cy.on('drag', 'node', (evt) => {
      constrainNodeToViewport(evt.target, cy);
    });

    syncScreenSpaceSizing(cy, renderMode, thumbSizeRef.current);
    notifyZoom();

    const tick = () => {
      if (!cy.destroyed()) {
        runForceStep(cy, attractorRef.current, dampingRef.current, selectedRef.current, velocityByIdRef.current);
      }
      simulationRef.current = requestAnimationFrame(tick);
    };
    simulationRef.current = requestAnimationFrame(tick);

    cyRef.current = cy;
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (simulationRef.current) cancelAnimationFrame(simulationRef.current);
      cy.destroy();
      cyRef.current = null;
    };
  }, [nodeData, renderMode]); // eslint-disable-line react-hooks/exhaustive-deps

  // EDGE SYNC — surgically replaces edges without touching node positions.
  useEffect(() => {
    linkDataRef.current = linkData;
    const cy = cyRef.current;
    if (!cy || cy.destroyed()) return;
    const nodeIdSet = new Set(cy.nodes().map((n) => n.id()));
    cy.edges().remove();
    const toAdd = (linkData || [])
      .map((e) => {
        const src = typeof e.source === 'object' ? e.source?.id : e.source;
        const tgt = typeof e.target === 'object' ? e.target?.id : e.target;
        if (!src || !tgt || !nodeIdSet.has(src) || !nodeIdSet.has(tgt)) return null;
        return { group: 'edges', data: { id: e.id, source: src, target: tgt, weight: e.weight, color: e.color } };
      })
      .filter(Boolean);
    if (toAdd.length) cy.add(toAdd);
  }, [linkData]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    syncScreenSpaceSizing(cy, renderMode, thumbnailSizePx);

    cy.nodes().unselect();
    if (selectedNodeId) {
      const node = cy.getElementById(selectedNodeId);
      if (node.length) {
        node.select();
      }
    }
  }, [selectedNodeId, renderMode, thumbnailSizePx]);

  return <div ref={containerRef} className="graph-canvas" />;
}