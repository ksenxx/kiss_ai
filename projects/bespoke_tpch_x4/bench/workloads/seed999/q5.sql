select n_name,  
    sum(l_extendedprice * (1 - l_discount)) as revenue 
FROM
    customer,  
    orders,  
    lineitem,  
    supplier,  
    nation,  
    region 
WHERE
    c_custkey = o_custkey 
    and l_orderkey = o_orderkey 
    and l_suppkey = s_suppkey 
    and c_nationkey = s_nationkey 
    and s_nationkey = n_nationkey 
    and n_regionkey = r_regionkey 
    and r_name = 'EUROPE' 
    and o_orderdate >= date '1994-01-01' 
    and o_orderdate < date '1994-01-01' + interval '1' year 
GROUP BY
    n_name 
ORDER BY
    revenue desc;