$cache = "$env:USERPROFILE\.gradle\caches\modules-2\files-2.1"
$tmp = "C:\Users\DracieVajko\AppData\Local\Temp\opencode"
$items = @(
  @{g="org.jetbrains.kotlinx"; a="kotlinx-coroutines-android"; v="1.11.0"; e="jar"},
  @{g="org.jetbrains.kotlinx"; a="kotlinx-coroutines-core-jvm"; v="1.11.0"; e="jar"}
)
foreach ($it in $items) {
  $base = "https://repo1.maven.org/maven2/$($it.g -replace '\.','/')/$($it.a)/$($it.v)"
  $u = "$base/$($it.a)-$($it.v).jar"
  curl.exe -sS --max-time 60 "$u.sha1" -o "$tmp\sha1.txt" 2>$null
  $sha = (Get-Content "$tmp\sha1.txt" -Raw).Trim()
  if (-not $sha -or $sha.Length -ne 40) { "MISSING sha1 for $($it.a)"; continue }
  $dir = "$cache\$($it.g)\$($it.a)\$($it.v)\$sha"
  New-Item -ItemType Directory -Force $dir | Out-Null
  $f = "$dir\$($it.a)-$($it.v).jar"
  if (-not (Test-Path $f)) { curl.exe -sS --max-time 300 -o $f $u }
  "{0} -> {1} bytes, sha1 ok" -f $it.a, (Get-Item $f).Length
}
