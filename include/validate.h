#ifndef VALIDATE_H_INCLUDED
#define VALIDATE_H_INCLUDED

#include "constants.h"
#include "king.h"
#include "magic.h"
#include "knight.h"
#include "pawn.h"
#include "util.h"

namespace validate {
    inline bool isValidPawnMove(moveInfo& currentMove, bool inCheck, gameState& current, U64* pieces, U64* occupied) {
        // called by isValidMove, so currentMove is up-to-date.

        if (currentMove.enPassant) {
            // enPassant capture.

            // check enPassant square.
            if (current.enPassantSquare != (int)(currentMove.finishSquare)) {
                return false;
            }

            // no need to check if captured piece present, guaranteed with ep square.
        } else {
            // regular move, capture or push.

            // check if finishSquare is empty or capturedPiece.
            if ((currentMove.capturedPieceType == 15 &&
                 (bool)((occupied[0] | occupied[1]) & (1ull << currentMove.finishSquare))) ||
                (currentMove.capturedPieceType != 15 &&
                 !(bool)(pieces[currentMove.capturedPieceType] & (1ull << currentMove.finishSquare)))) {
                return false;
            }

            // if double push, check that middle square is clear.
            if (abs((int)(currentMove.finishSquare) - (int)(currentMove.startSquare)) == 16) {
                if (currentMove.finishSquare > currentMove.startSquare) {
                    // white double push.
                    if ((occupied[0] | occupied[1]) & (1ull << (currentMove.startSquare + 8))) {
                        return false;
                    }
                } else {
                    // black double push.
                    if ((occupied[0] | occupied[1]) & (1ull << (currentMove.startSquare - 8))) {
                        return false;
                    }
                }
            }
        }

        // check if piece is pinned or if enPassant.
        U64 pinned = util::getPinnedPieces(currentMove.pieceType & 1, pieces, occupied);
        if ((pinned & (1ull << currentMove.startSquare)) || currentMove.enPassant || inCheck) {
            U64 start = 1ull << currentMove.startSquare;
            U64 finish = 1ull << currentMove.finishSquare;
            bool side = currentMove.pieceType & 1;
            pieces[currentMove.pieceType] -= start;
            pieces[currentMove.pieceType] += finish;
            occupied[(int)(side)] -= start;
            occupied[(int)(side)] += finish;
            if (currentMove.capturedPieceType != 15) {
                pieces[currentMove.capturedPieceType] -=
                    1ull << (currentMove.finishSquare + (int)(currentMove.enPassant) * (-8 + 16 * (side)));
                occupied[(int)(!side)] -=
                    1ull << (currentMove.finishSquare + (int)(currentMove.enPassant) * (-8 + 16 * (side)));
            }
            bool isBad = util::isInCheck(side, pieces, occupied);

            // unmove pieces.
            pieces[currentMove.pieceType] += start;
            pieces[currentMove.pieceType] -= finish;
            occupied[(int)(side)] += start;
            occupied[(int)(side)] -= finish;
            if (currentMove.capturedPieceType != 15) {
                pieces[currentMove.capturedPieceType] +=
                    1ull << (currentMove.finishSquare + (int)(currentMove.enPassant) * (-8 + 16 * (side)));
                occupied[(int)(!side)] +=
                    1ull << (currentMove.finishSquare + (int)(currentMove.enPassant) * (-8 + 16 * (side)));
            }
            if (isBad) {
                return false;
            }
        }

        return true;
    }

    inline bool isValidCastles(moveInfo& currentMove, bool inCheck, bool side, gameState& current, U64* pieces,
                               U64* occupied) {
        // called by isValidMove, so currentMove is up-to-date.
        if (inCheck) {
            return false;
        }

        // check that castling is allowed.
        if (currentMove.finishSquare - currentMove.startSquare == 2) {
            // kingside.
            if (!current.canKingCastle[(int)(side)]) {
                return false;
            }

            // castling squares not occupied or attacked.
            U64 attacked = util::updateAttacked(!side, pieces, occupied);
            U64 p = occupied[0] | occupied[1];
            if ((KING_CASTLE_OCCUPIED[(int)(side)] & p) || (KING_CASTLE_ATTACKED[(int)(side)] & attacked)) {
                return false;
            }
        } else {
            // queenside.
            if (!current.canQueenCastle[(int)(side)]) {
                return false;
            }

            // castling squares not occupied or attacked.
            U64 attacked = util::updateAttacked(!side, pieces, occupied);
            U64 p = occupied[0] | occupied[1];
            if ((QUEEN_CASTLE_OCCUPIED[(int)(side)] & p) || (QUEEN_CASTLE_ATTACKED[(int)(side)] & attacked)) {
                return false;
            }
        }

        return true;
    }

    inline bool isValidMove(U32 chessMove, bool inCheck, bool side, gameState& current, U64* pieces, U64* occupied) {
        // verifies if a move is valid in this position.
        // move is assumed to be legal from some other arbitrary position in the search tree.
        moveInfo currentMove = {
            .pieceType = (chessMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET,
            .startSquare = (chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET,
            .finishSquare = (chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET,
            .enPassant = (bool)(chessMove & MOVEINFO_ENPASSANT_MASK),
            .capturedPieceType = (chessMove & MOVEINFO_CAPTUREDPIECETYPE_MASK) >> MOVEINFO_CAPTUREDPIECETYPE_OFFSET,
            .finishPieceType = (chessMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET,
        };

        // check for correct side-to-move.
        if ((currentMove.pieceType & 1) != (side)) {
            return false;
        }

        // check that startSquare contains piece.
        if (!(bool)(pieces[currentMove.pieceType] & (1ull << currentMove.startSquare))) {
            return false;
        }

        // check for pawn move.
        if ((currentMove.pieceType >> 1) == (_nPawns >> 1)) {
            return isValidPawnMove(currentMove, inCheck, current, pieces, occupied);
        }
        // check for castles.
        if (((currentMove.pieceType >> 1) == (_nKing >> 1)) &&
            (abs((int)(currentMove.finishSquare) - (int)(currentMove.startSquare)) == 2)) {
            return isValidCastles(currentMove, inCheck, side, current, pieces, occupied);
        }

        // ordinary non-pawn capture/quiet.

        // check that finishSquare is empty or contains capturedPiece.
        if ((currentMove.capturedPieceType == 15 &&
             (bool)((occupied[0] | occupied[1]) & (1ull << currentMove.finishSquare))) ||
            (currentMove.capturedPieceType != 15 &&
             !(bool)(pieces[currentMove.capturedPieceType] & (1ull << currentMove.finishSquare)))) {
            return false;
        }

        // startSquare -> finishSquare is valid path for that piece.
        // knight moves are path legal.
        switch (currentMove.pieceType >> 1) {
        case _nKing >> 1:
            if (!(kingAttacks(pieces[currentMove.pieceType]) &
                  ~kingAttacks(pieces[_nKing + !(bool)(currentMove.pieceType & 1)]) &
                  (1ull << currentMove.finishSquare))) {
                return false;
            }
            break;
        case _nQueens >> 1:
            if (!(magicQueenAttacks(occupied[0] | occupied[1], currentMove.startSquare) &
                  (1ull << currentMove.finishSquare))) {
                return false;
            }
            break;
        case _nRooks >> 1:
            if (!(magicRookAttacks(occupied[0] | occupied[1], currentMove.startSquare) &
                  (1ull << currentMove.finishSquare))) {
                return false;
            }
            break;
        case _nBishops >> 1:
            if (!(magicBishopAttacks(occupied[0] | occupied[1], currentMove.startSquare) &
                  (1ull << currentMove.finishSquare))) {
                return false;
            }
            break;
        }

        // check if the piece to move is pinned and verify the move if necessary.
        U64 pinned = util::getPinnedPieces(currentMove.pieceType & 1, pieces, occupied);
        if ((pinned & (1ull << currentMove.startSquare)) || ((currentMove.pieceType >> 1) == (_nKing >> 1)) ||
            inCheck) {
            U64 start = 1ull << currentMove.startSquare;
            U64 finish = 1ull << currentMove.finishSquare;
            bool side = currentMove.pieceType & 1;
            pieces[currentMove.pieceType] -= start;
            pieces[currentMove.pieceType] += finish;
            occupied[(int)(side)] -= start;
            occupied[(int)(side)] += finish;
            if (currentMove.capturedPieceType != 15) {
                pieces[currentMove.capturedPieceType] -= finish;
                occupied[(int)(!side)] -= finish;
            }
            bool isBad = util::isInCheck(side, pieces, occupied);

            // unmove pieces.
            pieces[currentMove.pieceType] += start;
            pieces[currentMove.pieceType] -= finish;
            occupied[(int)(side)] += start;
            occupied[(int)(side)] -= finish;
            if (currentMove.capturedPieceType != 15) {
                pieces[currentMove.capturedPieceType] += finish;
                occupied[(int)(!side)] += finish;
            }
            if (isBad) {
                return false;
            }
        }

        // all checks passed!
        return true;
    }
} // namespace validate

#endif // VALIDATE_H_INCLUDED
