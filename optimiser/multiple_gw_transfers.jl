### A Pluto.jl notebook ###
# v0.20.15

using Markdown
using InteractiveUtils

# ╔═╡ d2ba2eab-2c55-4326-ab18-daba995220e8
begin
	using CSV, DataFrames, JuMP, HiGHS, PlutoUI, PrettyTables
end

# ╔═╡ 1bca1502-8031-11f0-3242-0d1043a241a4
md"# FPL Multi-Gameweek Optimiser with Transfers ⚽"

# ╔═╡ f193ee25-4deb-455d-9884-3211d5786cf2
# Pluto.jl width settings
html"""<style>
main {
	max-width: 80%;
	margin-left: 1%;
	margin-right: 10% !important;
}
"""

# ╔═╡ 20ca06b0-bbe8-4a89-b0cc-99487ff2fbe0
begin
	const START_GW = 14
	const NUM_GAMEWEEKS = 7
	
	const BUDGET = 100.6
	const GAMEWEEK_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
	const BENCH_WEIGHT = 0.1
	
	const SQUAD_SIZE = 15
	const STARTING_XI = 11
    const MAX_PLAYERS_PER_TEAM = 3
	const MAX_DEFENDERS_SAME_TEAM = 2

	const GK_REQUIRED = 2
    const DEF_REQUIRED = 5
    const MID_REQUIRED = 5
    const FWD_REQUIRED = 3
end;

# ╔═╡ 1980f18c-2054-4cfe-b07e-da8203cacd1b
begin
	END_GW = START_GW + NUM_GAMEWEEKS - 1
	df = CSV.read("../data/gw$(START_GW)-projection.csv", DataFrame)
	sort!(df, "$(START_GW)_Pts", rev=true)
end

# ╔═╡ 50a83055-a467-48b0-80dc-77c51d248248
function process_squad_data(squad_idx, starter_idx, captain_idx, df; gameweek=1)

    results = DataFrame(
        Name = String[],
        Pos = String[],
        Team = String[],
        Price = Float64[],
        xPts = Float64[]
    )
    
    pos_priority = Dict("G" => 1, "D" => 2, "M" => 3, "F" => 4)
    pts_col = "$(gameweek)_Pts"

	starters = [(df[i, "Name"], df[i, "Pos"], df[i, "Team"], df[i, "BV"], df[i, pts_col]) for i in starter_idx]

    bench_idx = setdiff(squad_idx, starter_idx)
    bench_players = [(df[i, "Name"], df[i, "Pos"], df[i, "Team"], df[i, "BV"], df[i, pts_col]) for i in bench_idx]
    
    sort!(starters, by = x -> (pos_priority[x[2]], -x[5]))
    sort!(bench_players, by = x -> (pos_priority[x[2]], -x[5]))
    
    for player in starters
        push!(results, player)
    end
    for player in bench_players
        push!(results, player)
    end
    
    captain_name = df[captain_idx, "Name"]
    bench_names = Set([df[i, "Name"] for i in bench_idx])
    
    return results, captain_name, bench_names
end

# ╔═╡ cadf55af-c61b-48d8-b292-72438005c856
function extract_transfer_plan(df, squad, starter, captain, transfer_in, transfer_out)
    n = nrow(df)
    squads_by_gw = Dict()
    transfers_by_gw = Dict()
    
    for gw in START_GW:END_GW
		squad_idx = findall(i -> value(squad[i, gw]) > 0.5, 1:n)
        starter_idx = findall(i -> value(starter[i, gw]) > 0.5, 1:n)
        captain_idx = findfirst(i -> value(captain[i, gw]) > 0.5, 1:n)
		
        squad_df, captain_name, bench_names = process_squad_data(squad_idx, starter_idx, captain_idx, df; gameweek=gw)
        squads_by_gw[gw] = (squad_df, captain_name, bench_names)
        
        if gw > START_GW
            transfers_in_idx = findall(i -> value(transfer_in[i, gw]) > 0.5, 1:n)
            transfers_out_idx = findall(i -> value(transfer_out[i, gw]) > 0.5, 1:n)
            
            transfers_in = ["$(df[i, "Name"]) ($(df[i, "Pos"]), $(df[i, "Team"]), £$(df[i, "BV"])m)" for i in transfers_in_idx]
            transfers_out = ["$(df[i, "Name"]) ($(df[i, "Pos"]), $(df[i, "Team"]), £$(df[i, "BV"])m)" for i in transfers_out_idx]
            
            transfers_by_gw[gw] = (transfers_in, transfers_out)
        end
    end
    
    return squads_by_gw, transfers_by_gw
end

# ╔═╡ 757049ba-7657-4cea-b5d9-1289809fd4c6
function optimise_multiple_gw_transfers(df)
	model = Model(HiGHS.Optimizer)
	n = nrow(df)

    @variable(model, squad[1:n, START_GW:END_GW], Bin)
    @variable(model, starter[1:n, START_GW:END_GW], Bin)
    @variable(model, captain[1:n, START_GW:END_GW], Bin)

	# Key differences: 
	# - Squad changes over time via transfers (1 per gameweek)
	# - Each gameweek has its own squad/lineup/captain
    @variable(model, transfer_in[1:n, (START_GW + 1):END_GW], Bin) 
    @variable(model, transfer_out[1:n, (START_GW + 1):END_GW], Bin)

    @objective(model, Max, sum(GAMEWEEK_WEIGHTS[gw - START_GW + 1] * df[i, "$(gw)_Pts"] * (starter[i, gw] + captain[i, gw] + BENCH_WEIGHT * (squad[i, gw] - starter[i, gw])) for i in 1:n, gw in START_GW:END_GW))

	# Transfer
    for gw in (START_GW + 1):END_GW
        @constraint(model, sum(transfer_in[i, gw] for i in 1:n) <= 1)
        @constraint(model, sum(transfer_out[i, gw] for i in 1:n) <= 1)
        @constraint(model, sum(transfer_in[i, gw] for i in 1:n) == sum(transfer_out[i, gw] for i in 1:n))
    end

	# Squad evolution
    for gw in (START_GW + 1):END_GW, i in 1:n
        @constraint(model, squad[i, gw] == squad[i, gw-1] + transfer_in[i, gw] - transfer_out[i, gw])
    end

	for gw in START_GW:END_GW
		# Budget
		@constraint(model, sum(df[i, "BV"] * squad[i, gw] for i in 1:n) <= BUDGET)
	
		# Squad composition
        @constraint(model, sum(squad[i, gw] for i in 1:n) == SQUAD_SIZE)
        @constraint(model, sum(starter[i, gw] for i in 1:n) == STARTING_XI)
        @constraint(model, sum(captain[i, gw] for i in 1:n) == 1)
        @constraint(model, sum(squad[i, gw] for i in 1:n if df[i, "Pos"] == "G") == GK_REQUIRED)
        @constraint(model, sum(squad[i, gw] for i in 1:n if df[i, "Pos"] == "D") == DEF_REQUIRED)
        @constraint(model, sum(squad[i, gw] for i in 1:n if df[i, "Pos"] == "M") == MID_REQUIRED)
        @constraint(model, sum(squad[i, gw] for i in 1:n if df[i, "Pos"] == "F") == FWD_REQUIRED)
	
		# Team limits
		for team in unique(df.Team)
			@constraint(model, sum(squad[i, gw] for i in 1:n if df[i, "Team"] == team) <= MAX_PLAYERS_PER_TEAM)
			@constraint(model, sum(squad[i, gw] for i in 1:n if df[i, "Team"] == team && (df[i, "Pos"] == "G" || df[i, "Pos"] == "D")) <= MAX_DEFENDERS_SAME_TEAM)
		end
	
		# Hierarchy
		for i in 1:n
            @constraint(model, starter[i, gw] <= squad[i, gw])
            @constraint(model, captain[i, gw] <= starter[i, gw])
		end
	
		# Formations
		@constraint(model, sum(starter[i, gw] for i in 1:n if df[i, "Pos"] == "G") == 1)
		@constraint(model, 3 <= sum(starter[i, gw] for i in 1:n if df[i, "Pos"] == "D") <= 5)
		@constraint(model, 2 <= sum(starter[i, gw] for i in 1:n if df[i, "Pos"] == "M") <= 5)
		@constraint(model, 1 <= sum(starter[i, gw] for i in 1:n if df[i, "Pos"] == "F") <= 3)
	end
	
	optimize!(model)

    return extract_transfer_plan(df, squad, starter, captain, transfer_in, transfer_out)
end

# ╔═╡ 09efab45-6fbd-49e0-ada9-a7cdbea18504
# ╠═╡ show_logs = false
squads_by_gw, transfers_by_gw = optimise_multiple_gw_transfers(df); 

# ╔═╡ 4505094b-14b2-44c8-8fbf-c88cf194bdde
function display_squad(results_df, captain_name, bench_names)

    captain_highlighter = Highlighter(
        (data, i, j) -> i > 1 && string(data[i, 1]) == captain_name,
        background = :green,
        bold = true
    )

    bench_highlighter = Highlighter(
        (data, i, j) -> i > 1 && string(data[i, 1]) in bench_names,
        background = :blue
    )

    pretty_table(results_df; highlighters = (captain_highlighter, bench_highlighter))
end

# ╔═╡ 09cc7b0f-2955-4c04-ba3a-360072d34146
function display_transfer_plan(squads_by_gw, transfers_by_gw)
    for gw in sort(collect(keys(squads_by_gw)))
        println("\n" * "="^60)
        println("GAMEWEEK $gw")
        println("="^60)
        
        # Transfers
        if gw > START_GW && haskey(transfers_by_gw, gw)
            transfers_in, transfers_out = transfers_by_gw[gw]
            
            if !isempty(transfers_in) || !isempty(transfers_out)
                println("\nTRANSFERS:")
                if !isempty(transfers_out)
                    println("OUT: $(join(transfers_out, ", "))")
                end
                if !isempty(transfers_in)
                    println("IN:  $(join(transfers_in, ", "))")
                end
                println()
            else
                println("\nTRANSFERS: None")
                println()
            end
        end

		# Squad
        squad_df, captain_name, bench_names = squads_by_gw[gw]
        display_squad(squad_df, captain_name, bench_names)
        
        # Expected Points
        total_points = sum(squad_df.xPts[1:11]) + 
                      (captain_name != "" ? squad_df[squad_df.Name .== captain_name, :xPts][1] : 0)
        println("\n Expected Points: $(round(total_points, digits=1))")
    end
end

# ╔═╡ 2061c3cd-b345-46aa-9fd3-ae8bf12e8e1e
display_transfer_plan(squads_by_gw, transfers_by_gw)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.7.0"
HiGHS = "~1.19.0"
JuMP = "~1.28.0"
PlutoUI = "~0.7.69"
PrettyTables = "~2.4.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "0413e197e769cdc8a34dfd925f196155a1276776"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "910febccb28d493032495b7009dce7d7f7aee554"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.0.1"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.HiGHS]]
deps = ["HiGHS_jll", "MathOptIIS", "MathOptInterface", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "3fae2ee8d6ea22009532d5919ff592fcbcab0ad9"
uuid = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
version = "1.19.0"

[[deps.HiGHS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "f56d423e3f583e26ceaef08a15a270b28723c89a"
uuid = "8fd58aa0-07eb-5a78-9b36-339c94fd15ea"
version = "1.11.0+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InlineStrings]]
git-tree-sha1 = "8594fac023c5ce1ef78260f24d1ad18b4327b420"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.4"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "d05a696a5abaf9d1f8bce948ee53ed1533fadfdb"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.28.0"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptIIS]]
deps = ["MathOptInterface"]
git-tree-sha1 = "31d4a6353ea00603104f11384aa44dd8b7162b28"
uuid = "8c4f8055-bd93-4160-a86b-a0c04941dbff"
version = "0.1.1"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON3", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "ce5a316de39941da67730ffec38e5396f7504853"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.42.1"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "491bdcdc943fcbc4c005900d7463c9f216aabf4c"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "2d7662f95eafd3b6c346acdbfc11a762a2256375"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.69"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═1bca1502-8031-11f0-3242-0d1043a241a4
# ╠═f193ee25-4deb-455d-9884-3211d5786cf2
# ╠═20ca06b0-bbe8-4a89-b0cc-99487ff2fbe0
# ╠═d2ba2eab-2c55-4326-ab18-daba995220e8
# ╠═1980f18c-2054-4cfe-b07e-da8203cacd1b
# ╠═757049ba-7657-4cea-b5d9-1289809fd4c6
# ╠═09efab45-6fbd-49e0-ada9-a7cdbea18504
# ╠═2061c3cd-b345-46aa-9fd3-ae8bf12e8e1e
# ╠═cadf55af-c61b-48d8-b292-72438005c856
# ╠═50a83055-a467-48b0-80dc-77c51d248248
# ╠═4505094b-14b2-44c8-8fbf-c88cf194bdde
# ╠═09cc7b0f-2955-4c04-ba3a-360072d34146
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
