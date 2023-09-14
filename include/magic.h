#ifndef MAGIC_H_INCLUDED
#define MAGIC_H_INCLUDED

#include "slider.h"

//b stores occupied squares (any piece type/colour).

typedef const unsigned long long C64;

static const U64 rookMasks[64]=
{
	C64(0x000101010101017E), C64(0x000202020202027C), C64(0x000404040404047A), C64(0x0008080808080876),
	C64(0x001010101010106E), C64(0x002020202020205E), C64(0x004040404040403E), C64(0x008080808080807E),
	C64(0x0001010101017E00), C64(0x0002020202027C00), C64(0x0004040404047A00), C64(0x0008080808087600),
	C64(0x0010101010106E00), C64(0x0020202020205E00), C64(0x0040404040403E00), C64(0x0080808080807E00),
	C64(0x00010101017E0100), C64(0x00020202027C0200), C64(0x00040404047A0400), C64(0x0008080808760800),
	C64(0x00101010106E1000), C64(0x00202020205E2000), C64(0x00404040403E4000), C64(0x00808080807E8000),
	C64(0x000101017E010100), C64(0x000202027C020200), C64(0x000404047A040400), C64(0x0008080876080800),
	C64(0x001010106E101000), C64(0x002020205E202000), C64(0x004040403E404000), C64(0x008080807E808000),
	C64(0x0001017E01010100), C64(0x0002027C02020200), C64(0x0004047A04040400), C64(0x0008087608080800),
	C64(0x0010106E10101000), C64(0x0020205E20202000), C64(0x0040403E40404000), C64(0x0080807E80808000),
	C64(0x00017E0101010100), C64(0x00027C0202020200), C64(0x00047A0404040400), C64(0x0008760808080800),
	C64(0x00106E1010101000), C64(0x00205E2020202000), C64(0x00403E4040404000), C64(0x00807E8080808000),
	C64(0x007E010101010100), C64(0x007C020202020200), C64(0x007A040404040400), C64(0x0076080808080800),
	C64(0x006E101010101000), C64(0x005E202020202000), C64(0x003E404040404000), C64(0x007E808080808000),
	C64(0x7E01010101010100), C64(0x7C02020202020200), C64(0x7A04040404040400), C64(0x7608080808080800),
	C64(0x6E10101010101000), C64(0x5E20202020202000), C64(0x3E40404040404000), C64(0x7E80808080808000)
};

const int rBits[64] = {
  12, 11, 11, 11, 11, 11, 11, 12,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  11, 10, 10, 10, 10, 10, 10, 11,
  12, 11, 11, 11, 11, 11, 11, 12
};

static const U64 rookMagics[64]={
    36029071906791552ull,
    1154047718063083552ull,
    45040394859186176ull,
    144141645376921604ull,
    216176080782889026ull,
    180144534884189185ull,
    74346811615084608ull,
    648518904689213696ull,
    4611756388246364160ull,
    6926580208451993668ull,
    36046406385926656ull,
    45669332186832952ull,
    650207780317429794ull,
    2252083318228993ull,
    576856585080016964ull,
    289361773627181504ull,
    18028142547451936ull,
    1152939131692778240ull,
    1298162626984939552ull,
    297259703111520256ull,
    74309471228133665ull,
    4630298621074473480ull,
    2305843627974459970ull,
    289392289359339648ull,
    17636209467968ull,
    4616260021561335808ull,
    2594082186828185664ull,
    1157451775982174720ull,
    4251433507554918528ull,
    1126213473009856ull,
    4513512417134744ull,
    78813131320607491ull,
    4620711363919495168ull,
    576480887111157776ull,
    4611694815059378180ull,
    594791819020605472ull,
    36033333059127296ull,
    720611133360177288ull,
    2378466963413270656ull,
    144115326590124160ull,
    4936509241331566592ull,
    147510480553394179ull,
    9007474671888388ull,
    345687014119571716ull,
    146652861265872928ull,
    1152943494856771592ull,
    300651627764844564ull,
    20275011636301826ull,
    72348964895330306ull,
    4612864695028678665ull,
    5765808361598091290ull,
    5297403042428093452ull,
    144150376743895572ull,
    853977012899856ull,
    36065081443762200ull,
    146651230452601344ull,
    1207341085106434ull,
    738590408161896577ull,
    281612453413897ull,
    4616752636795306498ull,
    4508006398034177ull,
    562984380346466ull,
    1179243505647617ull,
    569839240356098ull,
};

static const U64 bishopMasks[64]=
{
	C64(0x0040201008040200), C64(0x0000402010080400), C64(0x0000004020100A00), C64(0x0000000040221400),
	C64(0x0000000002442800), C64(0x0000000204085000), C64(0x0000020408102000), C64(0x0002040810204000),
	C64(0x0020100804020000), C64(0x0040201008040000), C64(0x00004020100A0000), C64(0x0000004022140000),
	C64(0x0000000244280000), C64(0x0000020408500000), C64(0x0002040810200000), C64(0x0004081020400000),
	C64(0x0010080402000200), C64(0x0020100804000400), C64(0x004020100A000A00), C64(0x0000402214001400),
	C64(0x0000024428002800), C64(0x0002040850005000), C64(0x0004081020002000), C64(0x0008102040004000),
	C64(0x0008040200020400), C64(0x0010080400040800), C64(0x0020100A000A1000), C64(0x0040221400142200),
	C64(0x0002442800284400), C64(0x0004085000500800), C64(0x0008102000201000), C64(0x0010204000402000),
	C64(0x0004020002040800), C64(0x0008040004081000), C64(0x00100A000A102000), C64(0x0022140014224000),
	C64(0x0044280028440200), C64(0x0008500050080400), C64(0x0010200020100800), C64(0x0020400040201000),
	C64(0x0002000204081000), C64(0x0004000408102000), C64(0x000A000A10204000), C64(0x0014001422400000),
	C64(0x0028002844020000), C64(0x0050005008040200), C64(0x0020002010080400), C64(0x0040004020100800),
	C64(0x0000020408102000), C64(0x0000040810204000), C64(0x00000A1020400000), C64(0x0000142240000000),
	C64(0x0000284402000000), C64(0x0000500804020000), C64(0x0000201008040200), C64(0x0000402010080400),
	C64(0x0002040810204000), C64(0x0004081020400000), C64(0x000A102040000000), C64(0x0014224000000000),
	C64(0x0028440200000000), C64(0x0050080402000000), C64(0x0020100804020000), C64(0x0040201008040200)
};

const int bBits[64] = {
  6, 5, 5, 5, 5, 5, 5, 6,
  5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 7, 7, 7, 7, 5, 5,
  5, 5, 7, 9, 9, 7, 5, 5,
  5, 5, 7, 9, 9, 7, 5, 5,
  5, 5, 7, 7, 7, 7, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5,
  6, 5, 5, 5, 5, 5, 5, 6
};

static const U64 bishopMagics[64]=
{
    54052133425123392ull,
    36150806904832ull,
    71748519462928ull,
    2308103713031716864ull,
    38852368850362368ull,
    594555003046199296ull,
    2306687850776625153ull,
    4612259997900935200ull,
    1152922124256559116ull,
    4611731102734747746ull,
    90177437499392ull,
    3458773998183127648ull,
    720647550402822148ull,
    576496486720733186ull,
    4793949099263008ull,
    4573977016081408ull,
    3171695359388942892ull,
    328762846533918848ull,
    72621111128494085ull,
    46179506221746178ull,
    145267484885327880ull,
    35192967422982ull,
    1876329959845272576ull,
    2306442243789166592ull,
    613615724686475520ull,
    4504462922493968ull,
    2312601157836605968ull,
    4828426157802194976ull,
    585758241435172868ull,
    596727226646134912ull,
    27059260349612548ull,
    2594372590018232580ull,
    299101527098498ull,
    18050222907597056ull,
    1197436899967232ull,
    2310390623668144640ull,
    290279659676160ull,
    180219439216857352ull,
    4611969710429570052ull,
    1441222803956175872ull,
    1157715384967627264ull,
    2305865583830566145ull,
    18025535368073217ull,
    405888120110717664ull,
    4900516729035163714ull,
    18120236305749040ull,
    2459038145157536010ull,
    2379026778181087872ull,
    4611690434274795528ull,
    288266111892324352ull,
    1152941845639155712ull,
    4683752984118427680ull,
    1235112207004598272ull,
    1729400974816643072ull,
    4630830749788939264ull,
    864776924722106688ull,
    146384863816196612ull,
    120490531624206848ull,
    17314619649ull,
    9024821506099200ull,
    280330567788ull,
    3458817565846407296ull,
    9011631795742848ull,
    38280768765640720ull,
};

static U64 rookMagicMoves[64][4096]={};
static U64 bishopMagicMoves[64][512]={};

U64 getRandomU64()
{
    U64 u1,u2,u3,u4;
    u1 = (U64)(rand()%65535) & 0xFFFF;
    u2 = (U64)(rand()%65535) & 0xFFFF;
    u3 = (U64)(rand()%65535) & 0xFFFF;
    u4 = (U64)(rand()%65535) & 0xFFFF;

    return u1 | (u2 << 16) | (u3 << 32) | (u4 << 48);
}

U64 getRandomU64FewBits()
{
    return getRandomU64() & getRandomU64() & getRandomU64();
}

U64 getBlocker(U64 mask, int index, int bits)
{
    U64 temp = mask;
    U64 b = 0;
    for (int i=0;i<bits;i++)
    {
        int lsb = popLSB(temp);
        if ((index >> i)%2 == 1)
        {
            b |= 1ull << (lsb);
        }
    }
    return b;
}

U64 getRookMagic(int square)
{
    U64 blockers[4096]; U64 attackSets[4096];

    //populate blockers and attack-sets.
    for (int i=0;i< (1 << rBits[square]);i++)
    {
        blockers[i] = getBlocker(rookMasks[square],i,rBits[square]);
        attackSets[i] = rookAttacks(1ull << (square),~blockers[i]);
    }

    U64 res = 0;

    for (int _=0;_<10000000;_++)
    {
        //get potential magicNum.
        U64 magic = getRandomU64FewBits();
        if (__builtin_popcountll((rookMasks[square]*magic) & 0xFF00000000000000ULL) < 6) {continue;}

        U64 magicMoves[4096]={};
        bool good=true;
        for (int i=0;i< (1 << rBits[square]);i++)
        {
            U64 x = blockers[i];
            x *= magic; x >>= 52;

            if (magicMoves[x] == 0ull) {magicMoves[x] = attackSets[i];}
            else if (magicMoves[x] != attackSets[i]) {good=false; break;}
        }
        if (good==true) {res = magic; break;}
    }
    return res;
}

U64 getBishopMagic(int square)
{
    U64 blockers[512]; U64 attackSets[512];

    //populate blockers and attack-sets.
    for (int i=0;i< (1 << bBits[square]);i++)
    {
        blockers[i] = getBlocker(bishopMasks[square],i,bBits[square]);
        attackSets[i] = bishopAttacks(1ull << (square),~blockers[i]);
    }

    U64 res = 0;

    for (int _=0;_<10000000;_++)
    {
        //get potential magicNum.
        U64 magic = getRandomU64FewBits();
        if (__builtin_popcountll((bishopMasks[square]*magic) & 0xFF00000000000000ULL) < 6) {continue;}

        U64 magicMoves[512]={};
        bool good=true;
        for (int i=0;i< (1 << bBits[square]);i++)
        {
            U64 x = blockers[i];
            x *= magic; x >>= 55;

            if (magicMoves[x] == 0ull) {magicMoves[x] = attackSets[i];}
            else if (magicMoves[x] != attackSets[i]) {good=false; break;}
        }
        if (good==true) {res = magic; break;}
    }
    return res;
}

void testRookMagics()
{
    for (int square=0;square<64;square++)
    {
        bool good = true;
        for (int i=0;i< (1 << rBits[square]);i++)
        {
            U64 b = getBlocker(rookMasks[square],i,rBits[square]);
            U64 check = rookAttacks(1ull << (square),~b);
            b *= rookMagics[square]; b >>= 52;
            U64 a = rookMagicMoves[square][b];
            if (a != check) {good=false; break;}
        }
        std::cout << square << " - " << good << std::endl;
    }
}

void testBishopMagics()
{
    for (int square=0;square<64;square++)
    {
        bool good = true;
        for (int i=0;i< (1 << bBits[square]);i++)
        {
            U64 b = getBlocker(bishopMasks[square],i,bBits[square]);
            U64 check = bishopAttacks(1ull << (square),~b);
            b *= bishopMagics[square]; b >>= 55;
            U64 a = bishopMagicMoves[square][b];
            if (a != check) {good=false; break;}
        }
        std::cout << square << " - " << good << std::endl;
    }
}

void populateMagicRookTables()
{
    for (int square=0;square<64;square++)
    {
        for (int i=0;i<(1 << rBits[square]);i++)
        {
            U64 b = getBlocker(rookMasks[square],i,rBits[square]);
            U64 attackSet = rookAttacks(1ull << (square),~b);
            b *= rookMagics[square]; b >>= 52;
            rookMagicMoves[square][b] = attackSet;
        }
    }
}

void populateMagicBishopTables()
{
    for (int square=0;square<64;square++)
    {
        for (int i=0;i<512;i++)
        {
            U64 b = getBlocker(bishopMasks[square],i,bBits[square]);
            U64 attackSet = bishopAttacks(1ull << (square),~b);
            b *= bishopMagics[square]; b >>= 55;
            bishopMagicMoves[square][b] = attackSet;
        }
    }
}

void populateMagicTables()
{
    populateMagicRookTables();
    populateMagicBishopTables();
}

inline U64 magicRookAttacks(const U64 b, const int square)
{
    return rookMagicMoves[square][((b & rookMasks[square]) * rookMagics[square]) >> 52ull];
}

inline U64 magicBishopAttacks(const U64 b, const int square)
{
    return bishopMagicMoves[square][((b & bishopMasks[square]) * bishopMagics[square]) >> 55ull];
}

inline U64 magicQueenAttacks(const U64 b, const int square)
{
    return (U64)(magicRookAttacks(b, square) | magicBishopAttacks(b, square));
}

#endif // MAGIC_H_INCLUDED
