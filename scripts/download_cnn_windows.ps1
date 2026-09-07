# Use Windows' certificate trust for the public binary downloads.
$ErrorActionPreference = 'Stop'
$cnnRepository = Split-Path -Parent $PSScriptRoot
$cnnDownloads = @(
    @{ Url = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'; Path = '.cache/cnn/vggface_vgg16.h5' },
    @{ Url = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v1.npy'; Path = '.cache/cnn/vggface_labels.npy' },
    @{ Url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'; Path = '.cache/keras/datasets/cifar-10-batches-py-target_archive' }
)
foreach ($cnnAsset in $cnnDownloads) {
    $cnnDestination = Join-Path $cnnRepository $cnnAsset.Path
    if (-not (Test-Path -LiteralPath $cnnDestination)) {
        $cnnPartial = $cnnDestination + '.partial'
        $cnnClient = New-Object System.Net.WebClient
        try {
            Write-Information -MessageData ('Downloading ' + $cnnAsset.Url) -InformationAction Continue
            $cnnClient.DownloadFile($cnnAsset.Url, $cnnPartial)
            Move-Item -LiteralPath $cnnPartial -Destination $cnnDestination
        }
        finally {
            $cnnClient.Dispose()
        }
    }
}
