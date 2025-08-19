pkgname=archvoice
pkgver=2.0.0
pkgrel=1
pkgdesc="Voice Changer на Python з GUI на PyQt6"
arch=('any')
url="https://github.com/HardWorkingMan-ua/ArchVoice"
license=('MIT')
depends=('python' 'python-numpy' 'python-sounddevice' 'python-scipy' 'python-pyqt6')
source=("voice_change.py" "archvoice.desktop" "icon.png")
sha256sums=('SKIP' 'SKIP' 'SKIP')

package() {
    # Сам скрипт
    install -Dm755 "$srcdir/voice_change.py" "$pkgdir/usr/share/archvoice/voice_change.py"

    # Створюємо папку для запускалки
    install -d "$pkgdir/usr/bin"

    # Запускалка
    echo '#!/bin/bash' > "$pkgdir/usr/bin/archvoice"
    echo 'exec python /usr/share/archvoice/voice_change.py "$@"' >> "$pkgdir/usr/bin/archvoice"
    chmod +x "$pkgdir/usr/bin/archvoice"

    # Іконка
    install -Dm644 "$srcdir/icon.png" "$pkgdir/usr/share/icons/hicolor/256x256/apps/archvoice.png"

    # .desktop файл
    install -Dm644 "$srcdir/archvoice.desktop" "$pkgdir/usr/share/applications/archvoice.desktop"
}
