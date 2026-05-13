import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import kotlin.io.path.Path
import kotlin.random.Random

fun main(baseSeed: Long): List<List<Int>> {
    val size = 256
    val sigma = 16.0

    val pairs = Array(size) { IntArray(size) }
    var i = 0
    for (y in 0..<size) {
        for (x in 0..<size) {
            pairs[y][x] = (i++) / 2
        }
    }

    val baseRandom = Random(baseSeed)
    val randoms = Array(size / 2) { Array(size / 2) { Random(baseRandom.nextLong()) } }

    fun sigmaToShuffleCount(sigma: Double): Int {
        return ((sigma * sigma) / 2.0 + 0.5).toInt()
    }

    fun shuffleGrid(offsetY: Int, offsetX: Int) {
        for (y in 0..<size / 2) {
            val dstY = y * 2 + offsetY
            for (x in 0..<size / 2) {
                val dstX = x * 2 + offsetX
                val permuteTemp = IntArray(4)
                var i = 0
                for (dy in 0..<2) {
                    for (dx in 0..<2) {
                        permuteTemp[i++] = pairs[(dstY + dy) % size][(dstX + dx) % size]
                    }
                }
                permuteTemp.shuffle(randoms[y][x])
                i = 0
                for (dy in 0..<2) {
                    for (dx in 0..<2) {
                        pairs[(dstY + dy) % size][(dstX + dx) % size] = permuteTemp[i++]
                    }
                }
            }
        }
    }

    repeat(sigmaToShuffleCount(sigma)) {
        shuffleGrid(it, it)
    }

    val pairPos = Array(size * size / 2) { IntArray(5) }
    for (y in 0..<size) {
        for (x in 0..<size) {
            val pairId = pairs[y][x]
            val arr = pairPos[pairId]
            val idx = (arr[0]++) * 2
            arr[idx + 1] = x
            arr[idx + 2] = y
        }
    }

    fun encodeMorton(x: Int, y: Int): Int {
        var res = x.toLong() or (y.toLong() shl 32)
        res = (res or (res shl 8)) and -0xf00ff
        res = (res or (res shl 4)) and -0x0f0f0f
        res = (res or (res shl 2)) and -0x3333333
        res = (res or (res shl 1)) and -0x5555555
        return (res or (res shr 31)).toInt()
    }

    pairPos.sortBy { minOf(encodeMorton(it[1], it[2]), encodeMorton(it[3], it[4])) }

    return pairPos.map { it.slice(1..<5) }
}

val baseRandom = Random(1145141919810L)
val basePath = Path("../shaders/textures")
repeat(8) {
    val data = main(baseRandom.nextLong())
    val outputPath = basePath.resolve("restir_reusetex${it}.bin")
    val outputData = ByteArray(data.size * 4)
    for (i in data.indices) {
        val pairData = data[i]
        val outputBase = i * 4
        outputData[outputBase] = (pairData[0] and 0xff).toByte()
        outputData[outputBase + 1] = (pairData[1] and 0xff).toByte()
        outputData[outputBase + 2] = (pairData[2] and 0xff).toByte()
        outputData[outputBase + 3] = (pairData[3] and 0xff).toByte()
    }
    outputPath.toFile().writeBytes(outputData)
}
