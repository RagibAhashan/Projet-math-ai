"""
    auc(gt::Array{<:Real}, scores::Array{<:Real})

Compute the area under the ROC curve based on the ground truth `gt` and the success probability `scores`.

See also `roc()` of MLBase.
"""
function auc(gt::Array{<:Real},scores::Array{<:Real})

    # Compute the ROC curve for 100 equally spaced thresholds - see `roc()`
    r = roc(gt, scores, 0:.01:1)

    # Compute the true positive rate and false positive rate
    tpr = true_positive_rate.(r)
    fpr = false_positive_rate.(r)

    # Numerical computation of the area under the ROC curve
    p = sortperm(fpr)

    permute!(tpr,p)
    permute!(fpr,p)

    area = 0.0

    for i in 2:length(tpr)
        dx = fpr[i] - fpr[i-1]
        dy = tpr[i] - tpr[i-1]
        area += dx*tpr[i-1] + dx*dy/2
    end

    return area

end


"""
    rocplot(gt::Array{<:Real},scores::Array{<:Real})

Show the ROC curve corresponding to the ground truth `gt` and the success probability `scores`.

The curve is computed for 100 equally spaced thresholds.
"""
function rocplot(gt::Array{<:Real},scores::Array{<:Real})

    # Compute the ROC curve for 100 equally spaced thresholds - see `roc()`
    r = roc(gt, scores, 0:.01:1)

    # Compute the true positive rate and false positive rate
    tpr = true_positive_rate.(r)
    fpr = false_positive_rate.(r)

    return plot(x=fpr, y=tpr, Geom.line, Geom.abline(color="red", style=:dash),
        Guide.xlabel("False Positive Rate"), Guide.ylabel("True Positive Rate"))

end

"""
    standardize!(X)
Standardisation des colonnes de la matrice (ou vecteur) X.
### Arguments
- `X` : Matrice des données.
### Details
La fonction modifie directement l'argument d'entrée X.
### Examples
\```
 julia> standardize!(X)
\```
"""
function standardize!(X::Array{T,2} where T<:Real)

    m = mean(X, dims=1)
    s = std(X, dims=1)

    n, p = size(X)

    for i=1:n
        for j=1:p
            X[i,j] = (X[i,j] - m[j]) / s[j]
        end
    end

    return X

end

function standardize!(X::Array{T,1} where T<:Real)

    m = mean(X)
    s = std(X)

    n = length(X)

    for i=1:n
        X[i] = (X[i] - m) ./ s
    end

    return X

end

"""
    standardize(X)
Standardisation des colonnes de la matrice (ou vecteur) X.
### Arguments
- `X` : Matrice des données.
### Details
La fonction renvoie une copie de la matrice standardisée.
### Examples
\```
 julia> Z = standardize(X)
\```
"""
function standardize(X::Array{T,2} where T<:Real)

    m = mean(X, dims=1)
    s = std(X, dims=1)

    n, p = size(X)

    Z = zeros(n, p)

    for i=1:n
        for j=1:p
            Z[i,j] = (X[i,j] - m[j]) / s[j]
        end
    end

    return Z

end

function standardize(X::Array{T,1} where T<:Real)

    m = mean(X)
    s = std(X)

    n = length(X)

    Z = zeros(n)

    for i=1:n
        Z[i] = (X[i] - m) ./ s
    end

    return Z

end

function compute_VIF(structureMatrix::Array{T,2} where T<:Real)

    n = size(structureMatrix,1)

    if all(isapprox.(structureMatrix[:,1], 1))
        m = size(structureMatrix,2)
        p = m-1  # nb de variables explicatives
        S = structureMatrix
    else
        p = size(structureMatrix,2)
        m = p+1
        S = hcat(ones(n), structureMatrix)
    end

    VIF = Float64[]

    for j in 2:m

        y = S[:,j]
        X = S[:, setdiff(1:m, j)]

        β̂ = X\y

        e = y - X*β̂

        SST = sum( (y .- mean(y)).^2)
        SSE = e'e

        R² = 1 - SSE/SST

        push!(VIF, 1/(1-R²))

    end

    return VIF

end
