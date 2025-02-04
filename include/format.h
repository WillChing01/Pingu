#ifndef FORMAT_H_INCLUDED
#define FORMAT_H_INCLUDED

#include "constants.h"
#include "bitboard.h"
#include "board.h"

#include <cctype>

const std::string promotionLetters = "_qrbn";
const std::string fileSymbols = "abcdefgh";
const std::string rankSymbols = "12345678";

bool isNumber(const std::string& input) {
    for (const auto& c : input) {
        if (!isdigit(c)) {
            return false;
        }
    }
    return true;
}

inline std::vector<std::string> separateByWhiteSpace(const std::string& input) {
    // assume only a single whitespace separates each word.
    std::vector<std::string> words;
    words.push_back("");
    for (const auto& character : input) {
        if (character == ' ') {
            words.push_back("");
        } else {
            words.back() += character;
        }
    }
    while (words.back() == "") {
        words.pop_back();
    }
    return words;
}

inline std::string moveToString(U32 chessMove) {
    std::string startSquare = toCoord((chessMove & MOVEINFO_STARTSQUARE_MASK) >> MOVEINFO_STARTSQUARE_OFFSET);
    std::string finishSquare = toCoord((chessMove & MOVEINFO_FINISHSQUARE_MASK) >> MOVEINFO_FINISHSQUARE_OFFSET);

    std::string res = startSquare + finishSquare;

    int pieceType = (chessMove & MOVEINFO_PIECETYPE_MASK) >> MOVEINFO_PIECETYPE_OFFSET;
    int finishPieceType = (chessMove & MOVEINFO_FINISHPIECETYPE_MASK) >> MOVEINFO_FINISHPIECETYPE_OFFSET;

    // check for promotion.
    if (pieceType != finishPieceType) {
        res += promotionLetters[finishPieceType >> 1];
    }

    return res;
}

inline U32 stringToMove(Board& b, const std::string& input) {
    // return 0 (null move) if input is not correct.
    if (input.length() != 4 && input.length() != 5) {
        return 0;
    }

    size_t startFile = fileSymbols.find(input[0]);
    size_t startRank = rankSymbols.find(input[1]);

    size_t finishFile = fileSymbols.find(input[2]);
    size_t finishRank = rankSymbols.find(input[3]);

    if (startFile == std::string::npos || startRank == std::string::npos || finishFile == std::string::npos ||
        finishRank == std::string::npos) {
        return 0;
    }

    U32 startSquare = 8 * startRank + startFile;
    U32 finishSquare = 8 * finishRank + finishFile;

    size_t finishPiece = 0;

    if (input.length() == 5) {
        finishPiece = promotionLetters.find(input[4]) * 2 + (b.side);
        if (finishPiece == std::string::npos) {
            return 0;
        }
    }

    // check if move is legal.
    b.generatePseudoMoves();

    for (int i = 0; i < (int)b.moveBuffer.size(); i++) {
        b.unpackMove(b.moveBuffer[i]);
        if (b.currentMove.startSquare != startSquare || b.currentMove.finishSquare != finishSquare) {
            continue;
        }
        if (b.currentMove.pieceType != b.currentMove.finishPieceType &&
            b.currentMove.finishPieceType != (U32)finishPiece) {
            continue;
        }
        return b.moveBuffer[i];
    }

    return 0;
}

inline std::string positionToFen(const U64* pieces, const gameState& current, bool side) {
    const std::string pieceTypes = "KkQqRrBbNnPp";
    std::string fen = "";

    // piece placement.
    for (int i = 7; i >= 0; i--) {
        int c = 0;
        for (int j = 0; j < 8; j++) {
            U64 square = 1ull << (8 * i + j);
            bool occ = false;
            for (int k = 0; k < 12; k++) {
                // check if occupied by piece.
                if (pieces[k] & square) {
                    if (c > 0) {
                        fen += std::to_string(c);
                    }
                    fen += pieceTypes[k];
                    c = 0;
                    occ = true;
                }
            }
            if (!occ) {
                c++;
            }
        }
        if (c > 0) {
            fen += std::to_string(c);
        }
        if (i > 0) {
            fen += '/';
        }
    }

    // side to move.
    fen += side & 1 ? " b" : " w";

    // castling rights.
    fen += ' ';
    if (current.canKingCastle[0] || current.canKingCastle[1] || current.canQueenCastle[0] ||
        current.canQueenCastle[1]) {
        if (current.canKingCastle[0]) {
            fen += 'K';
        }
        if (current.canQueenCastle[0]) {
            fen += 'Q';
        }
        if (current.canKingCastle[1]) {
            fen += 'k';
        }
        if (current.canQueenCastle[1]) {
            fen += 'q';
        }
    } else {
        fen += '-';
    }

    // en passant square.
    fen += ' ';
    fen += current.enPassantSquare != -1 ? toCoord(current.enPassantSquare) : "-";

    // move numbers.
    fen += " 0 1";

    return fen;
}

#endif // FORMAT_H_INCLUDED
