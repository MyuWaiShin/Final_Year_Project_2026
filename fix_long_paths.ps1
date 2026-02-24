# Enable Windows Long Path support (Requires Admin)
# If this fails, you might need to ask your Uni Admin to enable this.
try {
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    Write-Host "✓ Windows Long Paths Enabled Successfully." -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to enable Long Paths automatically (Requires Admin)." -ForegroundColor Red
    Write-Host "Attempting install without deep-nested tools..."
}
