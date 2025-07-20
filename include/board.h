#ifndef BOARD_H_INCLUDED
#define BOARD_H_INCLUDED

#include "constants.h"
#include "bitboard.h"
#include "king.h"
#include "knight.h"
#include "pawn.h"
#include "magic.h"
#include "util.h"
#include "killer.h"
#include "history.h"
#include "see.h"
#include "evaluation.h"
#include "nnue.h"
#include "transposition.h"

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

inline std::string positionToFen(const U64* pieces, const gameState& current, bool side);

class Board {
  public:
    U64 pieces[12] = {};

    U64 occupied[2] = {0, 0};
    bool side = 0;

    int startingPly = 1;
    int plyOffset = 0;

    std::vector<gameState> stateHistory;
    std::vector<U32> moveHistory;
    std::vector<U32> hashHistory;
    std::vector<int> irrevMoveInd;

    std::vector<U32> moveBuffer;
    std::vector<std::pair<U32, int>> moveCache[MAXDEPTH + 1 + 64] = {};
    std::vector<std::pair<U32, int>> badCaptures[MAXDEPTH + 1 + 64] = {};

    gameState current = {
        .canKingCastle = {true, true},
        .canQueenCastle = {true, true},
        .enPassantSquare = -1,
    };

    moveInfo currentMove = {};

    const int piecePhases[6] = {0, 4, 2, 1, 1, 0};

    int phase = 24;

    // overall zHash is XOR of these two.
    U64 zHashPieces = 0;
    U64 zHashState = 0;

    // modules.
    Killer killer;
    History history;
    SEE see;
    NNUE nnue;

    // temp variable for move appending.
    U32 newMove;

    Board() {
        // connect modules.
        see = SEE(pieces, occupied);
        nnue = NNUE(pieces, &side);

        // start position default.
        setPositionFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    };

    void copyBoard(Board& b) {
        for (int i = 0; i < 12; i++) {
            pieces[i] = b.pieces[i];
        }
        for (int i = 0; i < 2; i++) {
            occupied[i] = b.occupied[i];
        }
        side = b.side;

        stateHistory = b.stateHistory;
        moveHistory = b.moveHistory;
        hashHistory = b.hashHistory;
        irrevMoveInd = b.irrevMoveInd;

        current = b.current;
        phase = b.phase;
        zHashPieces = b.zHashPieces;
        zHashState = b.zHashState;

        nnue.fullRefresh();
    }

    void zHashHardUpdate() {
        zHashPieces = 0;
        zHashState = 0;

        for (int i = 0; i < 12; i++) {
            U64 temp = pieces[i];
            while (temp) {
                zHashPieces ^= randomNums[ZHASH_PIECES[i] + popLSB(temp)];
            }
        }

        if (side) {
            zHashPieces ^= randomNums[ZHASH_TURN];
        }

        if (current.enPassantSquare != -1) {
            zHashState ^= randomNums[ZHASH_ENPASSANT[current.enPassantSquare & 7]];
        }

        if (current.canKingCastle[0]) {
            zHashState ^= randomNums[ZHASH_CASTLES[0]];
        }
        if (current.canKingCastle[1]) {
            zHashState ^= randomNums[ZHASH_CASTLES[1]];
        }
        if (current.canQueenCastle[0]) {
            zHashState ^= randomNums[ZHASH_CASTLES[2]];
        }
        if (current.canQueenCastle[1]) {
            zHashState ^= randomNums[ZHASH_CASTLES[3]];
        }
    }

    void phaseHardUpdate() {
        phase = 0;
        for (int i = 0; i < 12; i++) {
            U64 temp = pieces[i];
            while (temp) {
                phase += piecePhases[i >> 1];
                popLSB(temp);
            }
        }
    }

    void setPositionFen(const std::string& fen) {
        // reset history.
        stateHistory.clear();
        moveHistory.clear();
        hashHistory.clear();
        irrevMoveInd.clear();

        std::vector<std::string> temp;
        temp.push_back("");

        for (int i = 0; i < (int)fen.length(); i++) {
            if (fen[i] == ' ') {
                temp.push_back("");
            } else {
                temp.back() += fen[i];
            }
        }

        // piece placement.
        for (int i = 0; i < 12; i++) {
            pieces[i] = 0;
        }

        U32 square = 56;
        for (int i = 0; i < (int)temp[0].length(); i++) {
            if (temp[0][i] == '/') {
                square -= 16;
            } else if ((int)(temp[0][i] - '0') < 9) {
                square += (int)(temp[0][i] - '0');
            } else if (temp[0][i] == 'K') {
                pieces[_nKing] += 1ull << square++;
            } else if (temp[0][i] == 'Q') {
                pieces[_nQueens] += 1ull << square++;
            } else if (temp[0][i] == 'R') {
                pieces[_nRooks] += 1ull << square++;
            } else if (temp[0][i] == 'B') {
                pieces[_nBishops] += 1ull << square++;
            } else if (temp[0][i] == 'N') {
                pieces[_nKnights] += 1ull << square++;
            } else if (temp[0][i] == 'P') {
                pieces[_nPawns] += 1ull << square++;
            } else if (temp[0][i] == 'k') {
                pieces[_nKing + 1] += 1ull << square++;
            } else if (temp[0][i] == 'q') {
                pieces[_nQueens + 1] += 1ull << square++;
            } else if (temp[0][i] == 'r') {
                pieces[_nRooks + 1] += 1ull << square++;
            } else if (temp[0][i] == 'b') {
                pieces[_nBishops + 1] += 1ull << square++;
            } else if (temp[0][i] == 'n') {
                pieces[_nKnights + 1] += 1ull << square++;
            } else if (temp[0][i] == 'p') {
                pieces[_nPawns + 1] += 1ull << square++;
            }
        }

        updateOccupied();

        // side to move.
        moveHistory.push_back(0);
        if (temp[1] == "w") {
            moveHistory.push_back(0);
            side = 0;
        } else {
            side = 1;
        }

        current = {
            .canKingCastle = {false, false},
            .canQueenCastle = {false, false},
            .enPassantSquare = -1,
        };

        // castling rights.
        for (int i = 0; i < (int)temp[2].length(); i++) {
            if (temp[2][i] == 'K') {
                current.canKingCastle[0] = true;
            } else if (temp[2][i] == 'k') {
                current.canKingCastle[1] = true;
            } else if (temp[2][i] == 'Q') {
                current.canQueenCastle[0] = true;
            } else if (temp[2][i] == 'q') {
                current.canQueenCastle[1] = true;
            }
        }

        // en passant square.
        if (temp[3] != "-") {
            current.enPassantSquare = toSquare(temp[3]);
        }

        startingPly = 2 * std::stoi(temp[5]) - !side;
        plyOffset = side ? -1 : -2;

        zHashHardUpdate();
        phaseHardUpdate();
        nnue.fullRefresh();

        // hash and state history.
        stateHistory.push_back(current);
        hashHistory.push_back(zHashPieces ^ zHashState);
        if (temp[1] == "w") {
            stateHistory.push_back(current);
            hashHistory.push_back(zHashPieces ^ zHashState);
        }
    }

    bool isCheckingMove(U32 chessMove) {
        // verifies if a legal move gives check.
        U32 pieceType = (chessMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
        U32 finishSquare = (chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;

        switch (pieceType >> 1) {
        case _nQueens >> 1:
            if (magicRookAttacks(occupied[0] | occupied[1], finishSquare) & pieces[_nKing + (int)(!side)]) {
                return true;
            }
            if (magicBishopAttacks(occupied[0] | occupied[1], finishSquare) & pieces[_nKing + (int)(!side)]) {
                return true;
            }
            break;
        case _nRooks >> 1:
            if (magicRookAttacks(occupied[0] | occupied[1], finishSquare) & pieces[_nKing + (int)(!side)]) {
                return true;
            }
            break;
        case _nBishops >> 1:
            if (magicBishopAttacks(occupied[0] | occupied[1], finishSquare) & pieces[_nKing + (int)(!side)]) {
                return true;
            }
            break;
        case _nKnights >> 1:
            if (knightAttacks(1ull << finishSquare) & pieces[_nKing + (int)(!side)]) {
                return true;
            }
            break;
        case _nPawns >> 1:
            if (pawnAttacks(1ull << finishSquare, side) & pieces[_nKing + (int)(!side)]) {
                return true;
            }
            break;
        }

        U32 startSquare = (chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
        int kingPos = __builtin_ctzll(pieces[_nKing + (int)(!side)]);

        // regular discovered check (rook/bishop rays).
        if ((magicRookAttacks(occupied[0] | occupied[1], kingPos) & (1ull << startSquare)) &&
            (magicRookAttacks((occupied[0] | occupied[1]) ^ (1ull << startSquare), kingPos) &
             (pieces[_nRooks + (int)(side)] | pieces[_nQueens + (int)(side)]))) {
            return true;
        }
        if ((magicBishopAttacks(occupied[0] | occupied[1], kingPos) & (1ull << startSquare)) &&
            (magicBishopAttacks((occupied[0] | occupied[1]) ^ (1ull << startSquare), kingPos) &
             (pieces[_nBishops + (int)(side)] | pieces[_nQueens + (int)(side)]))) {
            return true;
        }

        // enpassant discovered check.
        if (chessMove & MOVEINFO_ENPASSANT_MASK) {
            U32 enPassantSquare = finishSquare - 8 + 16 * side;
            U64 after = (occupied[0] | occupied[1]) ^ (1ull << startSquare) ^ (1ull << finishSquare) ^
                        (1ull << enPassantSquare);
            if ((magicRookAttacks(occupied[0] | occupied[1], kingPos) & (1ull << startSquare)) &&
                (magicRookAttacks(after, kingPos) & (pieces[_nRooks + (int)(side)] | pieces[_nQueens + (int)(side)]))) {
                return true;
            }
            if ((magicBishopAttacks(occupied[0] | occupied[1], kingPos) & (1ull << startSquare)) &&
                (magicBishopAttacks(after, kingPos) &
                 (pieces[_nBishops + (int)(side)] | pieces[_nQueens + (int)(side)]))) {
                return true;
            }
        }

        // castles discovered check.
        if (pieceType >> 1 == _nKing >> 1 && abs((int)(finishSquare) - (int)(startSquare)) == 2) {
            if (finishSquare > startSquare) {
                if (magicRookAttacks((occupied[0] | occupied[1]) ^ startSquare, KING_ROOK_SQUARE[side] - 2) &
                    pieces[_nKing + (int)(!side)]) {
                    return true;
                }
            } else {
                if (magicRookAttacks((occupied[0] | occupied[1]) ^ startSquare, QUEEN_ROOK_SQUARE[side] + 3) &
                    pieces[_nKing + (int)(!side)]) {
                    return true;
                }
            }
        }

        return false;
    }

    void appendPawnCapture(U32 pieceType, U32 startSquare, U32 finishSquare, bool enPassant, bool shouldCheck) {
        // pawn captures, promotion and enPassant.
        bool promotion = false;
        U32 capturedPieceType = 15;

        if (enPassant) {
            // enPassant.
            capturedPieceType = _nPawns + (U32)(!side);
        } else if ((finishSquare >> 3) == (U32)(7 - 7 * (side))) {
            // promotion.
            promotion = true;
            if (((1ull << finishSquare) & occupied[(int)(!side)]) != 0) {
                // check for captures on promotion.
                U64 x = 1ull << finishSquare;
                for (U32 i = _nQueens + (!side); i < 12; i += 2) {
                    if ((x & pieces[i]) != 0) {
                        capturedPieceType = i;
                        break;
                    }
                }
            }
        } else {
            // regular pawn capture.
            U64 x = 1ull << finishSquare;
            for (U32 i = _nQueens + (!side); i < 12; i += 2) {
                if ((x & pieces[i]) != 0) {
                    capturedPieceType = i;
                    break;
                }
            }
        }

        if (shouldCheck) {
            // check if move is legal (does not leave king in check).
            // move pieces.
            U64 start = 1ull << startSquare;
            U64 finish = 1ull << finishSquare;
            pieces[pieceType] -= start;
            pieces[pieceType] += finish;
            occupied[(int)(side)] -= start;
            occupied[(int)(side)] += finish;
            if (capturedPieceType != 15) {
                pieces[capturedPieceType] -= 1ull << (finishSquare + (int)(enPassant) * (-8 + 16 * (side)));
                occupied[(int)(!side)] -= 1ull << (finishSquare + (int)(enPassant) * (-8 + 16 * (side)));
            }
            bool isBad = util::isInCheck(side, pieces, occupied);

            // unmove pieces.
            pieces[pieceType] += start;
            pieces[pieceType] -= finish;
            occupied[(int)(side)] += start;
            occupied[(int)(side)] -= finish;
            if (capturedPieceType != 15) {
                pieces[capturedPieceType] += 1ull << (finishSquare + (int)(enPassant) * (-8 + 16 * (side)));
                occupied[(int)(!side)] += 1ull << (finishSquare + (int)(enPassant) * (-8 + 16 * (side)));
            }
            if (isBad) {
                return;
            }
        }

        newMove = (pieceType << MOVEINFO_PIECETYPE_OFFSET) | (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
                  (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) | (enPassant << MOVEINFO_ENPASSANT_OFFSET) |
                  (capturedPieceType << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
                  (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);

        if (promotion) {
            // promotion.
            for (U32 i = _nQueens + (side); i < _nPawns; i += 2) {
                newMove &= ~MOVEINFO_FINISHPIECETYPE_MASK;
                newMove |= i << MOVEINFO_FINISHPIECETYPE_OFFSET;
                moveBuffer.push_back(newMove);
            }
        } else {
            // append normally.
            moveBuffer.push_back(newMove);
        }
    }

    void appendCapture(U32 pieceType, U32 startSquare, U32 finishSquare, bool shouldCheck) {
        U32 capturedPieceType = 15;
        // regular capture, loop through to find victim.
        U64 x = 1ull << finishSquare;
        for (U32 i = _nQueens + (!side); i < 12; i += 2) {
            if ((x & pieces[i]) != 0) {
                capturedPieceType = i;
                break;
            }
        }

        if (shouldCheck) {
            // check if move is legal (does not leave king in check).
            // move pieces.
            U64 start = 1ull << startSquare;
            U64 finish = 1ull << finishSquare;
            pieces[pieceType] -= start;
            pieces[pieceType] += finish;
            pieces[capturedPieceType] -= finish;
            occupied[(int)(side)] -= start;
            occupied[(int)(side)] += finish;
            occupied[(int)(!side)] -= finish;
            bool isBad = util::isInCheck(side, pieces, occupied);

            // unmove pieces.
            pieces[pieceType] += start;
            pieces[pieceType] -= finish;
            pieces[capturedPieceType] += finish;
            occupied[(int)(side)] += start;
            occupied[(int)(side)] -= finish;
            occupied[(int)(!side)] += finish;

            if (isBad) {
                return;
            }
        }

        newMove = (pieceType << MOVEINFO_PIECETYPE_OFFSET) | (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
                  (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) | (false << MOVEINFO_ENPASSANT_OFFSET) |
                  (capturedPieceType << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) |
                  (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);

        moveBuffer.push_back(newMove);
    }

    void appendQuiet(U32 pieceType, U32 startSquare, U32 finishSquare, bool shouldCheck) {
        // never check castles.
        if (shouldCheck) {
            // check if move is legal (does not leave king in check).
            // move pieces.
            U64 start = 1ull << startSquare;
            U64 finish = 1ull << finishSquare;
            pieces[pieceType] -= start;
            pieces[pieceType] += finish;
            occupied[(int)(side)] -= start;
            occupied[(int)(side)] += finish;
            bool isBad = util::isInCheck(side, pieces, occupied);

            // unmove pieces.
            pieces[pieceType] += start;
            pieces[pieceType] -= finish;
            occupied[(int)(side)] += start;
            occupied[(int)(side)] -= finish;

            if (isBad) {
                return;
            }
        }

        newMove = (pieceType << MOVEINFO_PIECETYPE_OFFSET) | (startSquare << MOVEINFO_STARTSQUARE_OFFSET) |
                  (finishSquare << MOVEINFO_FINISHSQUARE_OFFSET) | (false << MOVEINFO_ENPASSANT_OFFSET) |
                  (15u << MOVEINFO_CAPTUREDPIECETYPE_OFFSET) | (pieceType << MOVEINFO_FINISHPIECETYPE_OFFSET);

        moveBuffer.push_back(newMove);
    }

    void updateOccupied() {
        occupied[0] = pieces[_nKing] | pieces[_nQueens] | pieces[_nRooks] | pieces[_nBishops] | pieces[_nKnights] |
                      pieces[_nPawns];
        occupied[1] = pieces[_nKing + 1] | pieces[_nQueens + 1] | pieces[_nRooks + 1] | pieces[_nBishops + 1] |
                      pieces[_nKnights + 1] | pieces[_nPawns + 1];
    }

    void display() {
        // display the current position in console.

        const char symbols[12] = {'K', 'k', 'Q', 'q', 'R', 'r', 'B', 'b', 'N', 'n', 'P', 'p'};

        std::vector<std::vector<std::string>> grid(8, std::vector<std::string>(8, "[ ]"));

        U64 x;
        for (int i = 0; i < 12; i++) {
            x = pieces[i];
            for (int j = 0; j < 64; j++) {
                if (x & 1) {
                    grid[j / 8][j % 8][1] = symbols[i];
                }
                x = x >> 1;
            }
        }

        for (int i = 7; i >= 0; i--) {
            for (int j = 0; j < 8; j++) {
                std::cout << grid[i][j];
            }
            std::cout << " " << i + 1 << std::endl;
        }
        std::cout << " A  B  C  D  E  F  G  H" << std::endl;

        std::cout << positionToFen(pieces, current, side) << std::endl;
    }

    void generateCaptures(int numChecks) {
        // regular captures.
        U32 pos;
        U64 x;
        U64 temp;
        U64 p = (occupied[0] | occupied[1]);

        // king.
        pos = __builtin_ctzll(pieces[_nKing + (int)(side)]);
        x = kingAttacks(pieces[_nKing + (int)(side)]) & ~kingAttacks(pieces[_nKing + (int)(!side)]) &
            occupied[(int)(!side)];
        while (x) {
            appendCapture(_nKing + (int)(side), pos, popLSB(x), true);
        }

        if (numChecks == 2) {
            return;
        }

        U64 target = numChecks == 1 ? util::getCheckPiece(side, pos, pieces, occupied) : occupied[(int)(!side)];
        U64 pinned = util::getPinnedPieces(side, pieces, occupied);

        // pawns.
        temp = pieces[_nPawns + (int)(side)];
        U64 pawnPosBoard;
        bool canPromote = side ? (bool)(((temp & RANK_2) >> 8) & (~p)) : (bool)(((temp & RANK_7) << 8) & (~p));
        if (!canPromote && !(pawnAttacks(temp, side) & target)) {
            temp = 0;
        }
        while (temp) {
            pos = popLSB(temp);
            pawnPosBoard = 1ull << pos;
            x = pawnAttacks(pawnPosBoard, side) & target;

            while (x) {
                appendPawnCapture(_nPawns + (int)(side), pos, popLSB(x), false, (bool)(pawnPosBoard & pinned));
            }

            // promotion by moving forward.
            if (!side) {
                x = ((pawnPosBoard & RANK_7) << 8) & (~p);
            } else {
                x = ((pawnPosBoard & RANK_2) >> 8) & (~p);
            }

            while (x) {
                appendPawnCapture(
                    _nPawns + (int)(side), pos, popLSB(x), false, (bool)(pawnPosBoard & pinned) || numChecks > 0);
            }
        }

        // enPassant.
        if (current.enPassantSquare != -1) {
            temp = pawnAttacks(1ull << current.enPassantSquare, !side) & pieces[_nPawns + (int)(side)];
            while (temp) {
                pos = popLSB(temp);
                appendPawnCapture(_nPawns + (int)(side), pos, current.enPassantSquare, true, true);
            }
        }

        // knights.
        temp = pieces[_nKnights + (int)(side)] & ~pinned;
        if (!(knightAttacks(temp) & target)) {
            temp = 0;
        }
        while (temp) {
            pos = popLSB(temp);
            x = knightAttacks(1ull << pos) & target;
            while (x) {
                appendCapture(_nKnights + (int)(side), pos, popLSB(x), false);
            }
        }

        // bishops.
        temp = pieces[_nBishops + (int)(side)];
        while (temp) {
            pos = popLSB(temp);
            x = magicBishopAttacks(p, pos) & target;
            while (x) {
                appendCapture(_nBishops + (int)(side), pos, popLSB(x), (bool)((1ull << pos) & pinned));
            }
        }

        // rook.
        temp = pieces[_nRooks + (int)(side)];
        while (temp) {
            pos = popLSB(temp);
            x = magicRookAttacks(p, pos) & target;
            while (x) {
                appendCapture(_nRooks + (int)(side), pos, popLSB(x), (bool)((1ull << pos) & pinned));
            }
        }

        // queen.
        temp = pieces[_nQueens + (int)(side)];
        while (temp) {
            pos = popLSB(temp);
            x = magicQueenAttacks(p, pos) & target;
            while (x) {
                appendCapture(_nQueens + (int)(side), pos, popLSB(x), (bool)((1ull << pos) & pinned));
            }
        }
    }

    void generateQuiets(int numChecks) {
        // regular moves.
        U32 pos;
        U64 x;
        U64 temp;
        U64 p = (occupied[0] | occupied[1]);

        // castling.
        if (numChecks == 0 && (current.canKingCastle[(int)(side)] || current.canQueenCastle[(int)(side)])) {
            U64 attacked = util::updateAttacked(!side, pieces, occupied);
            if (current.canKingCastle[(int)(side)] && !(bool)(KING_CASTLE_OCCUPIED[(int)(side)] & p) &&
                !(bool)(KING_CASTLE_ATTACKED[(int)(side)] & attacked)) {
                // kingside castle.
                pos = __builtin_ctzll(pieces[_nKing + (int)(side)]);
                appendQuiet(_nKing + (int)(side), pos, pos + 2, false);
            }
            if (current.canQueenCastle[(int)(side)] && !(bool)(QUEEN_CASTLE_OCCUPIED[(int)(side)] & p) &&
                !(bool)(QUEEN_CASTLE_ATTACKED[(int)(side)] & attacked)) {
                // queenside castle.
                pos = __builtin_ctzll(pieces[_nKing + (int)(side)]);
                appendQuiet(_nKing + (int)(side), pos, pos - 2, false);
            }
        }

        // king.
        pos = __builtin_ctzll(pieces[_nKing + (int)(side)]);
        x = kingAttacks(pieces[_nKing + (int)(side)]) & ~kingAttacks(pieces[_nKing + (int)(!side)]) & ~p;
        while (x) {
            appendQuiet(_nKing + (int)(side), pos, popLSB(x), true);
        }

        if (numChecks == 2) {
            return;
        }

        U64 mask = numChecks == 1 ? util::getBlockSquares(side, pos, pieces, occupied) : ~p;
        U64 pinned = util::getPinnedPieces(side, pieces, occupied);

        // knights.
        temp = pieces[_nKnights + (int)(side)] & ~pinned;
        while (temp) {
            pos = popLSB(temp);
            x = knightAttacks(1ull << pos) & mask;
            while (x) {
                appendQuiet(_nKnights + (int)(side), pos, popLSB(x), false);
            }
        }

        // bishops.
        temp = pieces[_nBishops + (int)(side)];
        while (temp) {
            pos = popLSB(temp);
            x = magicBishopAttacks(p, pos) & mask;
            while (x) {
                appendQuiet(_nBishops + (int)(side), pos, popLSB(x), (bool)((1ull << pos) & pinned));
            }
        }

        // pawns.
        temp = pieces[_nPawns + (int)(side)];
        U64 pawnPosBoard;
        while (temp) {
            pos = popLSB(temp);
            pawnPosBoard = 1ull << pos;
            x = 0;

            // move forward (exclude promotion).
            if (side == 0) {
                x |= ((pawnPosBoard & (~RANK_7)) << 8) & mask;
                x |= ((((pawnPosBoard & RANK_2) << 8) & (~p)) << 8) & mask;
            } else {
                x |= ((pawnPosBoard & (~RANK_2)) >> 8) & mask;
                x |= ((((pawnPosBoard & RANK_7) >> 8) & (~p)) >> 8) & mask;
            }

            while (x) {
                appendQuiet(_nPawns + (int)(side), pos, popLSB(x), (bool)(pawnPosBoard & pinned));
            }
        }

        // rook.
        temp = pieces[_nRooks + (int)(side)];
        while (temp) {
            pos = popLSB(temp);
            x = magicRookAttacks(p, pos) & mask;
            while (x) {
                appendQuiet(_nRooks + (int)(side), pos, popLSB(x), (bool)((1ull << pos) & pinned));
            }
        }

        // queen.
        temp = pieces[_nQueens + (int)(side)];
        while (temp) {
            pos = popLSB(temp);
            x = magicQueenAttacks(p, pos) & mask;
            while (x) {
                appendQuiet(_nQueens + (int)(side), pos, popLSB(x), (bool)((1ull << pos) & pinned));
            }
        }
    }

    bool generatePseudoMoves() {
        moveBuffer.clear();
        bool inCheck = util::isInCheck(side, pieces, occupied);
        U32 numChecks = 0;
        if (inCheck) {
            numChecks = util::isInCheckDetailed(side, pieces, occupied);
        }
        generateCaptures(numChecks);
        generateQuiets(numChecks);
        return inCheck;
    }

    void movePieces() {
        // remove piece from start square;
        pieces[currentMove.pieceType] -= 1ull << (currentMove.startSquare);
        zHashPieces ^= randomNums[64 * currentMove.pieceType + currentMove.startSquare];

        // add piece to end square, accounting for promotion.
        pieces[currentMove.finishPieceType] += 1ull << (currentMove.finishSquare);
        zHashPieces ^= randomNums[64 * currentMove.finishPieceType + currentMove.finishSquare];

        // update phase on promotion.
        if (currentMove.pieceType != currentMove.finishPieceType) {
            phase += piecePhases[currentMove.finishPieceType >> 1];
        }

        // remove any captured pieces.
        if (currentMove.capturedPieceType != 15) {
            int capturedSquare =
                currentMove.finishSquare + (int)(currentMove.enPassant) * (-8 + 16 * (currentMove.pieceType & 1));
            pieces[currentMove.capturedPieceType] -= 1ull << capturedSquare;
            zHashPieces ^= randomNums[64 * currentMove.capturedPieceType + capturedSquare];

            // update the game phase.
            phase -= piecePhases[currentMove.capturedPieceType >> 1];
        }

        // if castles, then move the rook too.
        if (currentMove.pieceType >> 1 == _nKing >> 1 &&
            abs((int)(currentMove.finishSquare) - (int)(currentMove.startSquare)) == 2) {
            if (currentMove.finishSquare - currentMove.startSquare == 2) {
                // kingside.
                pieces[_nRooks + (currentMove.pieceType & 1)] -= KING_ROOK_POS[currentMove.pieceType & 1];
                pieces[_nRooks + (currentMove.pieceType & 1)] += KING_ROOK_POS[currentMove.pieceType & 1] >> 2;

                zHashPieces ^= randomNums[64 * (_nRooks + (currentMove.pieceType & 1)) +
                                          KING_ROOK_SQUARE[currentMove.pieceType & 1]];
                zHashPieces ^= randomNums[64 * (_nRooks + (currentMove.pieceType & 1)) +
                                          KING_ROOK_SQUARE[currentMove.pieceType & 1] - 2];
            } else {
                // queenside.
                pieces[_nRooks + (currentMove.pieceType & 1)] -= QUEEN_ROOK_POS[currentMove.pieceType & 1];
                pieces[_nRooks + (currentMove.pieceType & 1)] += QUEEN_ROOK_POS[currentMove.pieceType & 1] << 3;

                zHashPieces ^= randomNums[64 * (_nRooks + (currentMove.pieceType & 1)) +
                                          QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]];
                zHashPieces ^= randomNums[64 * (_nRooks + (currentMove.pieceType & 1)) +
                                          QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3];
            }
        }

        updateOccupied();
    }

    void unMovePieces() {
        // remove piece from destination square.
        pieces[currentMove.finishPieceType] -= 1ull << (currentMove.finishSquare);
        zHashPieces ^= randomNums[64 * currentMove.finishPieceType + currentMove.finishSquare];

        // add piece to start square.
        pieces[currentMove.pieceType] += 1ull << (currentMove.startSquare);
        zHashPieces ^= randomNums[64 * currentMove.pieceType + currentMove.startSquare];

        // update phase on promotion.
        if (currentMove.pieceType != currentMove.finishPieceType) {
            phase -= piecePhases[currentMove.finishPieceType >> 1];
        }

        // add back captured pieces.
        if (currentMove.capturedPieceType != 15) {
            int capturedSquare =
                currentMove.finishSquare + (int)(currentMove.enPassant) * (-8 + 16 * (currentMove.pieceType & 1));
            pieces[currentMove.capturedPieceType] += 1ull << capturedSquare;
            zHashPieces ^= randomNums[64 * currentMove.capturedPieceType + capturedSquare];

            // update the game phase.
            phase += piecePhases[currentMove.capturedPieceType >> 1];
        }

        // if castles move the rook back.
        if (currentMove.pieceType >> 1 == _nKing >> 1 &&
            abs((int)(currentMove.finishSquare) - (int)(currentMove.startSquare)) == 2) {
            if (currentMove.finishSquare - currentMove.startSquare == 2) {
                // kingside.
                pieces[_nRooks + (currentMove.pieceType & 1)] -= KING_ROOK_POS[currentMove.pieceType & 1] >> 2;
                pieces[_nRooks + (currentMove.pieceType & 1)] += KING_ROOK_POS[currentMove.pieceType & 1];

                zHashPieces ^= randomNums[64 * (_nRooks + (currentMove.pieceType & 1)) +
                                          KING_ROOK_SQUARE[currentMove.pieceType & 1]];
                zHashPieces ^= randomNums[64 * (_nRooks + (currentMove.pieceType & 1)) +
                                          KING_ROOK_SQUARE[currentMove.pieceType & 1] - 2];
            } else {
                // queenside.
                pieces[_nRooks + (currentMove.pieceType & 1)] -= QUEEN_ROOK_POS[currentMove.pieceType & 1] << 3;
                pieces[_nRooks + (currentMove.pieceType & 1)] += QUEEN_ROOK_POS[currentMove.pieceType & 1];

                zHashPieces ^= randomNums[64 * (_nRooks + (currentMove.pieceType & 1)) +
                                          QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]];
                zHashPieces ^= randomNums[64 * (_nRooks + (currentMove.pieceType & 1)) +
                                          QUEEN_ROOK_SQUARE[currentMove.pieceType & 1] + 3];
            }
        }

        updateOccupied();
    }

    void unpackMove(U32 chessMove) {
        currentMove.pieceType = (chessMove & MOVEINFO_PIECETYPE_MASK);
        currentMove.startSquare = (chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
        currentMove.finishSquare = (chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
        currentMove.enPassant = (chessMove & MOVEINFO_ENPASSANT_MASK);
        currentMove.capturedPieceType =
            (chessMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
        currentMove.finishPieceType = (chessMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;
    }

    void makeMove(U32 chessMove) {
        unpackMove(chessMove);

        // save zHash.
        hashHistory.push_back(zHashPieces ^ zHashState);

        // move pieces.
        movePieces();

        // update history.
        stateHistory.push_back(current);
        moveHistory.push_back(chessMove);

        // turn increment can be done in zHashPieces.
        zHashPieces ^= randomNums[ZHASH_TURN];
        zHashState = 0;
        side = !side;

        // irrev move.
        if (currentMove.pieceType >> 1 == _nPawns >> 1 || currentMove.capturedPieceType != 15 ||
            (currentMove.pieceType >> 1 == _nKing >> 1 &&
             abs((int)currentMove.finishSquare - (int)currentMove.startSquare) == 2)) {
            irrevMoveInd.push_back(moveHistory.size() - 1);
        }

        // if double-pawn push, set en-passant square.
        // otherwise, set en-passant square to -1.
        if (currentMove.pieceType >> 1 == _nPawns >> 1 &&
            abs((int)(currentMove.finishSquare) - (int)(currentMove.startSquare)) == 16) {
            current.enPassantSquare = currentMove.finishSquare - 8 + 16 * (currentMove.pieceType & 1);
            zHashState ^= randomNums[ZHASH_ENPASSANT[current.enPassantSquare & 7]];
        } else {
            current.enPassantSquare = -1;
        }

        if (currentMove.pieceType >> 1 == _nRooks >> 1) {
            if (currentMove.startSquare == (U32)KING_ROOK_SQUARE[currentMove.pieceType & 1]) {
                current.canKingCastle[currentMove.pieceType & 1] = false;
            } else if (currentMove.startSquare == (U32)QUEEN_ROOK_SQUARE[currentMove.pieceType & 1]) {
                current.canQueenCastle[currentMove.pieceType & 1] = false;
            }
        } else if (currentMove.pieceType >> 1 == _nKing >> 1) {
            current.canKingCastle[currentMove.pieceType & 1] = false;
            current.canQueenCastle[currentMove.pieceType & 1] = false;
        }

        if (currentMove.capturedPieceType >> 1 == _nRooks >> 1) {
            if (currentMove.finishSquare == (U32)KING_ROOK_SQUARE[currentMove.capturedPieceType & 1]) {
                current.canKingCastle[currentMove.capturedPieceType & 1] = false;
            } else if (currentMove.finishSquare == (U32)QUEEN_ROOK_SQUARE[currentMove.capturedPieceType & 1]) {
                current.canQueenCastle[currentMove.capturedPieceType & 1] = false;
            }
        }

        // update castling rights for zHash.
        if (current.canKingCastle[0]) {
            zHashState ^= randomNums[ZHASH_CASTLES[0]];
        }
        if (current.canKingCastle[1]) {
            zHashState ^= randomNums[ZHASH_CASTLES[1]];
        }
        if (current.canQueenCastle[0]) {
            zHashState ^= randomNums[ZHASH_CASTLES[2]];
        }
        if (current.canQueenCastle[1]) {
            zHashState ^= randomNums[ZHASH_CASTLES[3]];
        }

        nnue.makeMove(chessMove);
    }

    void unmakeMove() {
        // unmake most recent move and update gameState.
        current = stateHistory.back();
        unpackMove(moveHistory.back());
        unMovePieces();

        // revert zhash for gamestate.
        zHashPieces ^= randomNums[ZHASH_TURN];
        zHashState = 0;
        side = !side;

        if (current.enPassantSquare != -1) {
            zHashState ^= randomNums[ZHASH_ENPASSANT[current.enPassantSquare & 7]];
        }

        if (current.canKingCastle[0]) {
            zHashState ^= randomNums[ZHASH_CASTLES[0]];
        }
        if (current.canKingCastle[1]) {
            zHashState ^= randomNums[ZHASH_CASTLES[1]];
        }
        if (current.canQueenCastle[0]) {
            zHashState ^= randomNums[ZHASH_CASTLES[2]];
        }
        if (current.canQueenCastle[1]) {
            zHashState ^= randomNums[ZHASH_CASTLES[3]];
        }

        nnue.unmakeMove(moveHistory.back());

        stateHistory.pop_back();
        moveHistory.pop_back();
        hashHistory.pop_back();

        if (irrevMoveInd.size() && irrevMoveInd.back() >= (int)moveHistory.size()) {
            irrevMoveInd.pop_back();
        }
    }

    void makeNullMove() {
        stateHistory.push_back(current);
        moveHistory.push_back(0);
        hashHistory.push_back(zHashPieces ^ zHashState);

        zHashPieces ^= randomNums[ZHASH_TURN];
        zHashState = 0;
        side = !side;

        irrevMoveInd.push_back(moveHistory.size() - 1);

        current.enPassantSquare = -1;

        if (current.canKingCastle[0]) {
            zHashState ^= randomNums[ZHASH_CASTLES[0]];
        }
        if (current.canKingCastle[1]) {
            zHashState ^= randomNums[ZHASH_CASTLES[1]];
        }
        if (current.canQueenCastle[0]) {
            zHashState ^= randomNums[ZHASH_CASTLES[2]];
        }
        if (current.canQueenCastle[1]) {
            zHashState ^= randomNums[ZHASH_CASTLES[3]];
        }
    }

    void unmakeNullMove() {
        current = stateHistory.back();

        zHashPieces ^= randomNums[ZHASH_TURN];
        zHashState = 0;
        side = !side;

        irrevMoveInd.pop_back();

        if (current.enPassantSquare != -1) {
            zHashState ^= randomNums[ZHASH_ENPASSANT[current.enPassantSquare & 7]];
        }

        if (current.canKingCastle[0]) {
            zHashState ^= randomNums[ZHASH_CASTLES[0]];
        }
        if (current.canKingCastle[1]) {
            zHashState ^= randomNums[ZHASH_CASTLES[1]];
        }
        if (current.canQueenCastle[0]) {
            zHashState ^= randomNums[ZHASH_CASTLES[2]];
        }
        if (current.canQueenCastle[1]) {
            zHashState ^= randomNums[ZHASH_CASTLES[3]];
        }

        stateHistory.pop_back();
        moveHistory.pop_back();
        hashHistory.pop_back();
    }

    int evaluateBoard() { return nnue.forward(); }

    void orderCaptures(int ply) {
        // order captures/promotions.
        moveCache[ply].clear();

        for (const auto& move : moveBuffer) {
            U32 capturedPieceType = (move & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET;
            U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            U32 finishPieceType = (move & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;

            int score = 32 * (15 - capturedPieceType) + pieceType;
            score += finishPieceType != pieceType ? (15 - finishPieceType) : 0;

            moveCache[ply].push_back(std::pair<U32, int>(move, score));
        }

        // sort the moves.
        std::sort(moveCache[ply].begin(), moveCache[ply].end(), [](auto& a, auto& b) { return a.second > b.second; });
    }

    void orderQuiets(int ply) {
        // order quiet moves by history.
        moveCache[ply].clear();

        for (const auto& move : moveBuffer) {
            U32 pieceType = (move & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
            U32 startSquare = (move & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET;
            U32 finishSquare = (move & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;

            int moveScore = history.scores[pieceType][finishSquare];
            if (moveHistory.size() && moveHistory.back() != 0) {
                U32 prevPieceType = (moveHistory.back() & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
                U32 prevFinishSquare =
                    (moveHistory.back() & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET;
                moveScore += 32 * history.extendedScores[prevPieceType][prevFinishSquare][pieceType >> 1][finishSquare];
            }

            if (pieceType & 1) {
                moveScore +=
                    PIECE_TABLES_START[pieceType >> 1][finishSquare] - PIECE_TABLES_START[pieceType >> 1][startSquare];
            } else {
                moveScore += PIECE_TABLES_START[pieceType >> 1][finishSquare ^ 56] -
                             PIECE_TABLES_START[pieceType >> 1][startSquare ^ 56];
            }

            moveCache[ply].push_back(std::pair<U32, int>(move, moveScore));
        }

        // sort the moves.
        std::sort(moveCache[ply].begin(), moveCache[ply].end(), [](auto& a, auto& b) { return a.second > b.second; });
    }
};

#endif // BOARD_H_INCLUDED
