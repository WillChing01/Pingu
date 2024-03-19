#ifndef MOVEGEN_H_INCLUDED
#define MOVEGEN_H_INCLUDED

#include "board.h"

void Board::resetCaptureCounter()
{
    return;
}

void Board::updateCaptureCounter()
{
    return;
}

void Board::generateNextCapture()
{
    if (!cc.attackerBB || !cc.victimBB) {updateCaptureCounter();}

    //check if finished.
    if (false) {return;}

    //append the move for next attacker.
    while (true)
    {
        continue;
    }
}

#endif // MOVEGEN_H_INCLUDED
