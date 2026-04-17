param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$TopK = 24,
    [int]$MaxEdges = 180,
    [double]$MinScore = 0.02,
    [string]$OutCsv = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutCsv)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutCsv = "data/ttls/panel_query_regression_${timestamp}.csv"
}

$queries = @(
    "waterfront vegetation topography",
    "river embankment floodplain planting",
    "contour map terrain slope section",
    "facade elevation building envelope",
    "masterplan circulation public space",
    "botanical species canopy grove planting",
    "bridge crossing waterfront infrastructure",
    "section perspective axonometric diagram",
    "historical restoration masonry structure",
    "competition board layout orientation"
)

function Get-RelationBreakdown {
    param(
        [object[]]$Edges
    )

    if (-not $Edges -or $Edges.Count -eq 0) {
        return ""
    }

    $counts = @{}
    foreach ($edge in $Edges) {
        $types = @()
        if ($null -ne $edge.connection_types) {
            $types = @($edge.connection_types)
        }

        if ($types.Count -eq 0) {
            $types = @("unknown")
        }

        foreach ($t in $types) {
            $key = [string]$t
            if (-not $counts.ContainsKey($key)) {
                $counts[$key] = 0
            }
            $counts[$key] += 1
        }
    }

    $ordered = $counts.GetEnumerator() | Sort-Object -Property Value -Descending
    return (($ordered | ForEach-Object { "{0}:{1}" -f $_.Key, $_.Value }) -join ";")
}

$results = @()

for ($i = 0; $i -lt $queries.Count; $i++) {
    $query = $queries[$i]
    Write-Host ("[{0}/{1}] Query: {2}" -f ($i + 1), $queries.Count, $query)

    $body = @{
        query = $query
        top_k = $TopK
        max_edges = $MaxEdges
        min_score = $MinScore
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/api/panel-query-graph" -Method Post -ContentType "application/json" -Body $body

        $nodes = @()
        $edges = @()
        $legendKeys = @()

        if ($null -ne $response.graph) {
            if ($null -ne $response.graph.nodes) {
                $nodes = @($response.graph.nodes)
            }
            if ($null -ne $response.graph.edges) {
                $edges = @($response.graph.edges)
            }
            if ($null -ne $response.graph.connection_color_map) {
                $legendKeys = @($response.graph.connection_color_map.PSObject.Properties.Name)
            }
        }

        $results += [PSCustomObject]@{
            query = $query
            ok = [bool]$response.ok
            matched = [int]$response.matched
            nodes = $nodes.Count
            edges = $edges.Count
            legend_types = ($legendKeys -join ",")
            relation_breakdown = Get-RelationBreakdown -Edges $edges
            error = ""
        }
    }
    catch {
        $results += [PSCustomObject]@{
            query = $query
            ok = $false
            matched = 0
            nodes = 0
            edges = 0
            legend_types = ""
            relation_breakdown = ""
            error = $_.Exception.Message
        }
    }
}

$outDir = Split-Path -Parent $OutCsv
if (-not [string]::IsNullOrWhiteSpace($outDir) -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

$results | Export-Csv -Path $OutCsv -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Regression summary"
$results | Format-Table -AutoSize

$successCount = ($results | Where-Object { $_.ok -eq $true }).Count
$nonEmptyCount = ($results | Where-Object { $_.nodes -gt 0 -and $_.edges -gt 0 }).Count

Write-Host ""
Write-Host ("Passed ok=true: {0}/{1}" -f $successCount, $results.Count)
Write-Host ("Non-empty graph: {0}/{1}" -f $nonEmptyCount, $results.Count)
Write-Host ("CSV written to: {0}" -f $OutCsv)