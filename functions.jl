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
function standardize2!(X::Array{T,2} where T<:Real)

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

function standardize2!(X::Array{T,1} where T<:Real)

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
function standardize2(X::Array{T,2} where T<:Real)

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

function standardize2(X::Array{T,1} where T<:Real)

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

"""
    GMM(ω::Real, μ₀::Real, σ₀::Real, μ₁::Real, σ₁::Real)

Création d'un objet de type `UnivariateMixture` de la librairie *Distributions.jl* ayant comme densité

```math
f(y) = (1-ω) ~ \\mathcal{N}( y\\mid μ₀, σ₀²) + ω ~ \\mathcal{N}( y\\mid μ₁, σ₁²)
```
"""
function GMM(ω::Real, μ₀::Real, σ₀::Real, μ₁::Real, σ₁::Real)
    
    pd = MixtureModel(Normal[ Normal(μ₀, σ₀), Normal(μ₁, σ₁)], [1-ω, ω])
    
    return pd
    
end


"""
    componentprob(mixturemodel::UnivariateMixture, y::Real; componentindex=1, logprob=false)

Calcul de la probabilité que y provienne de la composante `componentindex` du mélange `mixturemodel`.
"""
function componentprob(mixturemodel::UnivariateMixture, y::Real; componentindex=1, logprob=false)

    fc = component(mixturemodel,componentindex)
    
    lp = log(probs(mixturemodel)[componentindex]) + logpdf(fc,y) - logpdf(mixturemodel, y)
    
    if logprob
        return lp
    else
        return exp(lp)
    end
    
end

"""
    _emstep(pd::MixtureModel,y)

Réalisation d'une itération de l'algorithme EM à partir du mélange `pd` avec les données `y`.

#### Détails
La fonction met à jour les paramètres de la distribution `pd` avec les estimations améliorées.
"""
function _emstep(pd::MixtureModel,y)
    
    n = length(y)
    
    f₁ = component(pd, 2)
    ω = probs(pd)[2]
    
    lp₁ = log(ω) .+ logpdf.(f₁,y) - logpdf.(pd, y)
    p₁ = exp.(lp₁)
    
    ω̂ = sum(p₁)/n
    
    p₀ = 1 .- p₁
    
    μ̂₀ = sum( p₀.* y) / sum(p₀)
    
    σ̂₀² = sum( p₀.* (y .- μ̂₀).^2 ) / sum(p₀)
    
    μ̂₁ = sum( p₁.* y) / sum(p₁)
    
    σ̂₁² = sum( p₁.* (y .- μ̂₁).^2 ) / sum(p₁)
    
    fd = GMM(ω̂, μ̂₀, sqrt(σ̂₀²), μ̂₁, sqrt(σ̂₁²))
    
    return fd
    
end

"""
    GMMemfit(y::Vector{<:Real} ; initialvalue::Vector{<:Real}=Float64[], maxiter::Int=1000, tol::Real=2*eps())

Calcul des estimateurs du maximum de la vraisemblance d'un mélange de lois normales avec l'algorithme EM.
"""
function GMMemfit(y::Vector{<:Real} ; initialvalue::Vector{<:Real}=Float64[], maxiter::Int=1000, tol::Real=2*eps())
    
    if isempty(initialvalue)
        
        n = length(y)
        
        ind = (1:n) .< n/2
        
        y₀ = y[ind]
        y₁ = y[.!(ind)]
        
        initialvalue = [.5, mean(y₀), std(y₀), mean(y₁), std(y₁)]
        
    end
    
    pd = GMM(initialvalue...)
    
    iter = 1
    err = 1
    
    while (err > tol) & (iter < maxiter)
       
        fd = _emstep(pd,y)
        
        err = abs(loglikelihood(fd,y) - loglikelihood(pd,y))
        
        pd = fd
        
        iter +=1
        
    end
    
    μ₀ = mean(components(pd)[1])
    μ₁ = mean(components(pd)[2])

    if μ₀ > μ₁
        μ₀ = mean(components(pd)[2])
        σ₀ = std(components(pd)[2])
        μ₁ = mean(components(pd)[1])
        σ₁ = std(components(pd)[1])
        ω = probs(pd)[1]

        pd = GMM(ω, μ₀, σ₀, μ₁, σ₁)

    end
    
    
    
    if iter == maxiter
        println("Convergence not reached in $maxiter iterations")
    else
        println("Convergence reached in $iter iterations")
    end
    
 return pd
    
end

"""
    histplot(fd::UnivariateMixture, y::Vector{<:Real})

Trace le mélange de lois `fd` superposé à l'histogramme des données `y`.
"""
function histplot(fd::UnivariateMixture, y::Vector{<:Real})
   
    @assert length(components(fd)) == 2 "the function is optimized for a mixture of two components."
    
    nbin = floor(Int,sqrt(length(y)))
    opacity = repeat([0.75, 0.85], outer=nbin)
    
    xmin = minimum(y)
    xmax = maximum(y)
       
    plot(Guide.ylabel("densité"), Guide.xlabel("y"), Coord.cartesian(xmin=xmin, xmax=xmax),
        layer(x -> pdf(fd, x), xmin , xmax, Theme(default_color="black")),
        layer(x -> probs(fd)[1]*pdf(components(fd)[1], x), xmin , xmax, Geom.line, Theme(default_color="gold2")),
        layer(x -> probs(fd)[2]*pdf(components(fd)[2], x), xmin , xmax, Theme(default_color="red")),
        layer(x=y, alpha=opacity, Geom.histogram(position=:identity, bincount = nbin, density=true)),
    )
        
end
