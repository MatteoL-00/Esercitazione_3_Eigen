#include <iostream>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

int main() {
    // Definisci le matrici A e i vettori b per i tre sistemi
    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
           8.320502943378437e-01, -9.992887623566787e-01;

    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
           8.320502943378437e-01, -8.324762492991313e-01;

    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
           8.320502943378437e-01, -8.320502947645361e-01;

    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    // Risolvi i sistemi lineari utilizzando la decomposizione PALU
    cout << "Risoluzione dei sistemi lineari con decomposizione PALU:" << endl;
    Vector2d x1_pal = A1.partialPivLU().solve(b1);
    Vector2d x2_pal = A2.partialPivLU().solve(b2);
    Vector2d x3_pal = A3.partialPivLU().solve(b3);

    // Risolvi i sistemi lineari utilizzando la decomposizione QR
    cout << "Risoluzione dei sistemi lineari con decomposizione QR:" << endl;
    Vector2d x1_qr = A1.householderQr().solve(b1);
    Vector2d x2_qr = A2.householderQr().solve(b2);
    Vector2d x3_qr = A3.householderQr().solve(b3);

    // Stampa le soluzioni
    cout << "Sistema 1:" << endl;
    cout << "Soluzione PALU: " << x1_pal.transpose() << endl;
    cout << "Soluzione QR: " << x1_qr.transpose() << endl;

    cout << "Sistema 2:" << endl;
    cout << "Soluzione PALU: " << x2_pal.transpose() << endl;
    cout << "Soluzione QR: " << x2_qr.transpose() << endl;

    cout << "Sistema 3:" << endl;
    cout << "Soluzione PALU: " << x3_pal.transpose() << endl;
    cout << "Soluzione QR: " << x3_qr.transpose() << endl;

    // Calcola l'errore relativo
    Vector2d sol_esatta;
    sol_esatta << -1.0, -1.0;

    double errore_pal1 = (x1_pal - sol_esatta).norm() / sol_esatta.norm();
    double errore_qr1 = (x1_qr - sol_esatta).norm() / sol_esatta.norm();

    double errore_pal2 = (x2_pal - sol_esatta).norm() / sol_esatta.norm();
    double errore_qr2 = (x2_qr - sol_esatta).norm() / sol_esatta.norm();

    double errore_pal3 = (x3_pal - sol_esatta).norm() / sol_esatta.norm();
    double errore_qr3 = (x3_qr - sol_esatta).norm() / sol_esatta.norm();

    cout << "Errore relativo Sistema 1 PALU: " << errore_pal1 << endl;
    cout << "Errore relativo Sistema 1 QR: " << errore_qr1 << endl;

    cout << "Errore relativo Sistema 2 PALU: " << errore_pal2 << endl;
    cout << "Errore relativo Sistema 2 QR: " << errore_qr2 << endl;

    cout << "Errore relativo Sistema 3 PALU: " << errore_pal3 << endl;
    cout << "Errore relativo Sistema 3 QR: " << errore_qr3 << endl;

    return 0;
}
