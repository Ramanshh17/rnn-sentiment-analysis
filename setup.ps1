# Create directories
$directories = @(
    "src",
    "src\models",
    "src\data",
    "src\utils",
    "experiments",
    "notebooks",
    "checkpoints"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "✓ Created: $dir" -ForegroundColor Green
}

# Create __init__.py files
$initFiles = @(
    "src\__init__.py",
    "src\models\__init__.py",
    "src\data\__init__.py",
    "src\utils\__init__.py"
)

foreach ($file in $initFiles) {
    New-Item -ItemType File -Force -Path $file | Out-Null
    Write-Host "✓ Created: $file" -ForegroundColor Green
}

Write-Host "`n✓ Project structure created successfully!" -ForegroundColor Cyan