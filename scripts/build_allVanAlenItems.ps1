$ErrorActionPreference = 'Stop'

function Escape-TurtleLiteral([string]$value) {
    if ($null -eq $value) { return '' }
    $v = $value -replace '\\', '\\\\'
    $v = $v -replace '"', '`"'
    $v = $v -replace "`r`n|`n|`r", ' '
    return $v.Trim()
}

function To-Slug([string]$value) {
    if ([string]::IsNullOrWhiteSpace($value)) { return '' }
    $s = $value.ToLowerInvariant()
    $s = [regex]::Replace($s, '[^a-z0-9]+', '-')
    $s = [regex]::Replace($s, '-{2,}', '-')
    $s = $s.Trim('-')
    return $s
}

$items = Get-Content -Raw allitems.json | ConvertFrom-Json
$outPath = 'allVanAlenItems.ttl'
$lines = New-Object System.Collections.Generic.List[string]

$lines.Add('@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .')
$lines.Add('@prefix owl:     <http://www.w3.org/2002/07/owl#> .')
$lines.Add('@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .')
$lines.Add('@prefix dcterms: <http://purl.org/dc/terms/> .')
$lines.Add('@prefix archdrw: <https://archivingresearch.org/schema/drawing#> .')
$lines.Add('@prefix vai:     <https://archivingresearch.org/resource/van-alen/> .')
$lines.Add('')
$lines.Add('<https://archivingresearch.org/data/allVanAlenItems>')
$lines.Add('    a owl:Ontology ;')
$lines.Add('    dcterms:title "Van Alen Archive Item Graph"@en ;')
$lines.Add('    dcterms:description "RDF graph derived from allitems.json using Dublin Core Terms and the Archival Drawing ontology extension."@en ;')
$lines.Add('    dcterms:source "allitems.json" ;')
$lines.Add('    owl:imports <http://purl.org/dc/terms/> , <https://archivingresearch.org/schema/drawing> .')
$lines.Add('')

for ($i = 0; $i -lt $items.Count; $i++) {
    $item = $items[$i]

    $id = if ($null -ne $item.id) { [string]$item.id } else { '' }
    $filename = if ($null -ne $item.filename) { [string]$item.filename } else { '' }
    $title = if ($null -ne $item.title) { [string]$item.title } else { '' }
    $displayTitle = if ($null -ne $item.displayTitle) { [string]$item.displayTitle } else { '' }
    $year = $item.year
    $page = $item.page
    $type = if ($null -ne $item.type) { [string]$item.type } else { '' }
    $projectKey = if ($null -ne $item.projectKey) { [string]$item.projectKey } else { '' }
    $url = if ($null -ne $item.url) { [string]$item.url } else { '' }

    $slugSource = if (-not [string]::IsNullOrWhiteSpace($filename)) { $filename } elseif (-not [string]::IsNullOrWhiteSpace($id)) { $id } else { "item-$i" }
    $slug = To-Slug("$slugSource-$i")
    if ([string]::IsNullOrWhiteSpace($slug)) { $slug = "item-$i" }

    $class = if ($type -eq 'text') { 'archdrw:ArchivalText' } else { 'archdrw:ArchivalDrawing' }

    $predicates = New-Object System.Collections.Generic.List[string]
    $predicates.Add("a $class")

    $effectiveId = if (-not [string]::IsNullOrWhiteSpace($id)) { $id } else { "item-$('{0:d5}' -f $i)" }
    $predicates.Add("archdrw:instanceId `"$(Escape-TurtleLiteral $effectiveId)`"")

    if (-not [string]::IsNullOrWhiteSpace($title)) {
        $predicates.Add("dcterms:title `"$(Escape-TurtleLiteral $title)`"")
    }
    if ((-not [string]::IsNullOrWhiteSpace($displayTitle)) -and ($displayTitle -ne $title)) {
        $predicates.Add("dcterms:alternative `"$(Escape-TurtleLiteral $displayTitle)`"")
    }
    if (-not [string]::IsNullOrWhiteSpace($type)) {
        $predicates.Add("dcterms:type `"$(Escape-TurtleLiteral $type)`"")
    }
    if ($null -ne $year -and "$year" -match '^\d{4}$') {
        $predicates.Add("dcterms:date `"$year`"^^xsd:gYear")
    }
    if ($null -ne $page -and "$page" -match '^\d+$') {
        $predicates.Add("archdrw:boardPage `"$page`"^^xsd:integer")
    }
    if (-not [string]::IsNullOrWhiteSpace($projectKey)) {
        $pk = Escape-TurtleLiteral $projectKey
        $predicates.Add("archdrw:competition `"$pk`"")
        $predicates.Add("archdrw:projectKey `"$pk`"")
        $predicates.Add("dcterms:relation `"$pk`"")
    }
    if (-not [string]::IsNullOrWhiteSpace($filename)) {
        $fn = Escape-TurtleLiteral $filename
        $predicates.Add("archdrw:sourceFileName `"$fn`"")
        $predicates.Add("dcterms:identifier `"$fn`"")
        if ($filename.ToLowerInvariant().EndsWith('.jpg') -or $filename.ToLowerInvariant().EndsWith('.jpeg')) {
            $predicates.Add('dcterms:format "image/jpeg"')
        }
    } elseif (-not [string]::IsNullOrWhiteSpace($effectiveId)) {
        $predicates.Add("dcterms:identifier `"$(Escape-TurtleLiteral $effectiveId)`"")
    }
    if (-not [string]::IsNullOrWhiteSpace($url)) {
        $predicates.Add("dcterms:source `"$(Escape-TurtleLiteral $url)`"")
    }
    if ($null -ne $item.tags) {
        foreach ($tag in $item.tags) {
            $tagText = [string]$tag
            if (-not [string]::IsNullOrWhiteSpace($tagText)) {
                $predicates.Add("dcterms:subject `"$(Escape-TurtleLiteral $tagText)`"")
            }
        }
    }

    $lines.Add("vai:$slug")
    for ($p = 0; $p -lt $predicates.Count; $p++) {
        $suffix = if ($p -eq $predicates.Count - 1) { ' .' } else { ' ;' }
        $lines.Add("    $($predicates[$p])$suffix")
    }
    $lines.Add('')
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines((Join-Path (Get-Location) $outPath), $lines, $utf8NoBom)
"WROTE=$outPath"
"RECORDS=$($items.Count)"


