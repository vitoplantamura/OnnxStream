//
// This code is from TensorFlow.js:
//
// https://raw.githubusercontent.com/tensorflow/tfjs/master/tfjs-core/src/backends/non_max_suppression_impl.ts
//

function nonMaxSuppressionV3Impl(
    boxes,
    scores,
    maxOutputSize,
    iouThreshold,
    scoreThreshold
) {
    return nonMaxSuppressionImpl_(
        boxes,
        scores,
        maxOutputSize,
        iouThreshold,
        scoreThreshold,
        0 /* softNmsSigma */
    )
}

function nonMaxSuppressionImpl_(
    boxes,
    scores,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
    softNmsSigma,
    returnScoresTensor = false,
    padToMaxOutputSize = false,
    returnValidOutputs = false
) {
    // The list is sorted in ascending order, so that we can always pop the
    // candidate with the largest score in O(1) time.
    const candidates = []

    for (let i = 0; i < scores.length; i++) {
        if (scores[i] > scoreThreshold) {
            candidates.push({ score: scores[i], boxIndex: i, suppressBeginIndex: 0 })
        }
    }

    candidates.sort(ascendingComparator)

    // If softNmsSigma is 0, the outcome of this algorithm is exactly same as
    // before.
    const scale = softNmsSigma > 0 ? -0.5 / softNmsSigma : 0.0

    const selectedIndices = []
    const selectedScores = []

    while (selectedIndices.length < maxOutputSize && candidates.length > 0) {
        const candidate = candidates.pop()
        const { score: originalScore, boxIndex, suppressBeginIndex } = candidate

        if (originalScore < scoreThreshold) {
            break
        }

        // Overlapping boxes are likely to have similar scores, therefore we
        // iterate through the previously selected boxes backwards in order to
        // see if candidate's score should be suppressed. We use
        // suppressBeginIndex to track and ensure a candidate can be suppressed
        // by a selected box no more than once. Also, if the overlap exceeds
        // iouThreshold, we simply ignore the candidate.
        let ignoreCandidate = false
        for (let j = selectedIndices.length - 1; j >= suppressBeginIndex; --j) {
            const iou = intersectionOverUnion(boxes, boxIndex, selectedIndices[j])

            if (iou >= iouThreshold) {
                ignoreCandidate = true
                break
            }

            candidate.score =
                candidate.score * suppressWeight(iouThreshold, scale, iou)

            if (candidate.score <= scoreThreshold) {
                break
            }
        }

        // At this point, if `candidate.score` has not dropped below
        // `scoreThreshold`, then we know that we went through all of the
        // previous selections and can safely update `suppressBeginIndex` to the
        // end of the selected array. Then we can re-insert the candidate with
        // the updated score and suppressBeginIndex back in the candidate list.
        // If on the other hand, `candidate.score` has dropped below the score
        // threshold, we will not add it back to the candidates list.
        candidate.suppressBeginIndex = selectedIndices.length

        if (!ignoreCandidate) {
            // Candidate has passed all the tests, and is not suppressed, so
            // select the candidate.
            if (candidate.score === originalScore) {
                selectedIndices.push(boxIndex)
                selectedScores.push(candidate.score)
            } else if (candidate.score > scoreThreshold) {
                // Candidate's score is suppressed but is still high enough to be
                // considered, so add back to the candidates list.
                binaryInsert(candidates, candidate, ascendingComparator)
            }
        }
    }

    // NonMaxSuppressionV4 feature: padding output to maxOutputSize.
    const validOutputs = selectedIndices.length
    const elemsToPad = maxOutputSize - validOutputs

    if (padToMaxOutputSize && elemsToPad > 0) {
        selectedIndices.push(...new Array(elemsToPad).fill(0))
        selectedScores.push(...new Array(elemsToPad).fill(0.0))
    }

    const result = { selectedIndices }

    if (returnScoresTensor) {
        result["selectedScores"] = selectedScores
    }

    if (returnValidOutputs) {
        result["validOutputs"] = validOutputs
    }

    return result
}

function intersectionOverUnion(boxes, i, j) {
    const iCoord = boxes.subarray(i * 4, i * 4 + 4)
    const jCoord = boxes.subarray(j * 4, j * 4 + 4)
    const yminI = Math.min(iCoord[0], iCoord[2])
    const xminI = Math.min(iCoord[1], iCoord[3])
    const ymaxI = Math.max(iCoord[0], iCoord[2])
    const xmaxI = Math.max(iCoord[1], iCoord[3])
    const yminJ = Math.min(jCoord[0], jCoord[2])
    const xminJ = Math.min(jCoord[1], jCoord[3])
    const ymaxJ = Math.max(jCoord[0], jCoord[2])
    const xmaxJ = Math.max(jCoord[1], jCoord[3])
    const areaI = (ymaxI - yminI) * (xmaxI - xminI)
    const areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ)
    if (areaI <= 0 || areaJ <= 0) {
        return 0.0
    }
    const intersectionYmin = Math.max(yminI, yminJ)
    const intersectionXmin = Math.max(xminI, xminJ)
    const intersectionYmax = Math.min(ymaxI, ymaxJ)
    const intersectionXmax = Math.min(xmaxI, xmaxJ)
    const intersectionArea =
        Math.max(intersectionYmax - intersectionYmin, 0.0) *
        Math.max(intersectionXmax - intersectionXmin, 0.0)
    return intersectionArea / (areaI + areaJ - intersectionArea)
}

// A Gaussian penalty function, this method always returns values in [0, 1].
// The weight is a function of similarity, the more overlap two boxes are, the
// smaller the weight is, meaning highly overlapping boxe will be significantly
// penalized. On the other hand, a non-overlapping box will not be penalized.
function suppressWeight(iouThreshold, scale, iou) {
    const weight = Math.exp(scale * iou * iou)
    return iou <= iouThreshold ? weight : 0.0
}

function ascendingComparator(c1, c2) {
    // For objects with same scores, we make the object with the larger index go
    // first. In an array that pops from the end, this means that the object with
    // the smaller index will be popped first. This ensures the same output as
    // the TensorFlow python version.
    return (
        c1.score - c2.score || (c1.score === c2.score && c2.boxIndex - c1.boxIndex)
    )
}

/**
* Inserts a value into a sorted array. This method allows duplicate, meaning it
* allows inserting duplicate value, in which case, the element will be inserted
* at the lowest index of the value.
* @param arr The array to modify.
* @param element The element to insert.
* @param comparator Optional. If no comparator is specified, elements are
* compared using array_util.defaultComparator, which is suitable for Strings
* and Numbers in ascending arrays. If the array contains multiple instances of
* the target value, the left-most instance will be returned. To provide a
* comparator, it should take 2 arguments to compare and return a negative,
* zero, or a positive number.
*/
function binaryInsert(arr, element, comparator) {
    const index = binarySearch(arr, element, comparator)
    const insertionPoint = index < 0 ? -(index + 1) : index
    arr.splice(insertionPoint, 0, element)
}

/**
 * Searches the array for the target using binary search, returns the index
 * of the found element, or position to insert if element not found. If no
 * comparator is specified, elements are compared using array_
 * util.defaultComparator, which is suitable for Strings and Numbers in
 * ascending arrays. If the array contains multiple instances of the target
 * value, the left-most instance will be returned.
 * @param arr The array to be searched in.
 * @param target The target to be searched for.
 * @param comparator Should take 2 arguments to compare and return a negative,
 *    zero, or a positive number.
 * @return Lowest index of the target value if found, otherwise the insertion
 *    point where the target should be inserted, in the form of
 *    (-insertionPoint - 1).
 */
function binarySearch(arr, target, comparator) {
    return binarySearch_(arr, target, comparator || defaultComparator)
}

/**
 * Compares its two arguments for order.
 * @param a The first element to be compared.
 * @param b The second element to be compared.
 * @return A negative number, zero, or a positive number as the first
 *     argument is less than, equal to, or greater than the second.
 */
function defaultComparator(a, b) {
    return a > b ? 1 : a < b ? -1 : 0
}

function binarySearch_(arr, target, comparator) {
    let left = 0
    let right = arr.length
    let middle = 0
    let found = false
    while (left < right) {
        middle = left + ((right - left) >>> 1)
        const compareResult = comparator(target, arr[middle])
        if (compareResult > 0) {
            left = middle + 1
        } else {
            right = middle
            // If compareResult is 0, the value is found. We record it is found,
            // and then keep looking because there may be duplicate.
            found = !compareResult
        }
    }

    return found ? left : -left - 1
}
