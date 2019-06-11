#include "SharedMatting.h"

int main() {
    char fileAddr[64] = {0};

    for (int n = 1; n < 65; n++) {
    	SharedMatting sm;

	    sprintf(fileAddr, "input/input%d%d.jpg", n / 10, n % 10);
	    sm.LoadImage(fileAddr);

	    sprintf(fileAddr, "mask/mask%d%d.jpg", n / 10, n % 10);
	    sm.LoadMask(fileAddr);
	    
	    sprintf(fileAddr, "trimap/trimap%d%d.png", n / 10, n % 10);
	    sm.SolveAlpha(fileAddr);

	    sprintf(fileAddr, "result/result%d%d.png", n / 10, n % 10);
	    sm.SaveMatte(fileAddr);
    }
    
    return 0;
}
