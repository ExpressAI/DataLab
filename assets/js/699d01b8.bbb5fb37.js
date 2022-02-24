"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[829],{3905:function(e,t,a){a.d(t,{Zo:function(){return d},kt:function(){return f}});var r=a(7294);function n(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function o(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,r)}return a}function i(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?o(Object(a),!0).forEach((function(t){n(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):o(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function s(e,t){if(null==e)return{};var a,r,n=function(e,t){if(null==e)return{};var a,r,n={},o=Object.keys(e);for(r=0;r<o.length;r++)a=o[r],t.indexOf(a)>=0||(n[a]=e[a]);return n}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)a=o[r],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(n[a]=e[a])}return n}var c=r.createContext({}),l=function(e){var t=r.useContext(c),a=t;return e&&(a="function"==typeof e?e(t):i(i({},t),e)),a},d=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},p=r.forwardRef((function(e,t){var a=e.components,n=e.mdxType,o=e.originalType,c=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),p=l(a),f=n,m=p["".concat(c,".").concat(f)]||p[f]||u[f]||o;return a?r.createElement(m,i(i({ref:t},d),{},{components:a})):r.createElement(m,i({ref:t},d))}));function f(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var o=a.length,i=new Array(o);i[0]=p;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:n,i[1]=s;for(var l=2;l<o;l++)i[l]=a[l];return r.createElement.apply(null,i)}return r.createElement.apply(null,a)}p.displayName="MDXCreateElement"},6419:function(e,t,a){a.r(t),a.d(t,{frontMatter:function(){return s},contentTitle:function(){return c},metadata:function(){return l},toc:function(){return d},default:function(){return p}});var r=a(7462),n=a(3366),o=(a(7294),a(3905)),i=["components"],s={},c="Compare Two Datasets",l={unversionedId:"WebUI/compare_two_datasets",id:"WebUI/compare_two_datasets",title:"Compare Two Datasets",description:"When doing research, knowing the detailed differences between two datasets is important in multiple aspects, for example",source:"@site/docs/WebUI/6_compare_two_datasets.md",sourceDirName:"WebUI",slug:"/WebUI/compare_two_datasets",permalink:"/DataLab/docs/WebUI/compare_two_datasets",editUrl:"https://github.com/ExpressAI/DataLab/tree/main/docs/WebUI/6_compare_two_datasets.md",tags:[],version:"current",sidebarPosition:6,frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Identify Dataset Artifacts",permalink:"/DataLab/docs/WebUI/bias_analysis_for_artifacts"},next:{title:"Dataset Recommendation",permalink:"/DataLab/docs/WebUI/dataset_recommendation_based_on_idea"}},d=[{value:"1. Choose two datasets first",id:"1-choose-two-datasets-first",children:[],level:3},{value:"2. Click the right mouse button and choose the <code>compare</code>",id:"2-click-the-right-mouse-button-and-choose-the-compare",children:[],level:3},{value:"3. DataLab will generate different analytical figures and tables for these two datasets.",id:"3-datalab-will-generate-different-analytical-figures-and-tables-for-these-two-datasets",children:[],level:3}],u={toc:d};function p(e){var t=e.components,a=(0,n.Z)(e,i);return(0,o.kt)("wrapper",(0,r.Z)({},u,a,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"compare-two-datasets"},"Compare Two Datasets"),(0,o.kt)("p",null,"When doing research, knowing the detailed differences between two datasets is important in multiple aspects, for example"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"it can help us explain the diverse behaviors of models training on different datasets"),(0,o.kt)("li",{parentName:"ul"},"it can help us to choose a suitable one under a specific scenario\nHowever, analyzing their differences is tedious work, which usually needs to design different features and calculate them w.r.t to the datasets.")),(0,o.kt)("p",null,"DataLab automates this process and helps researchers to make pair-wise dataset analysis in a very convenient way. The general workflow is shown below:"),(0,o.kt)("h3",{id:"1-choose-two-datasets-first"},"1. Choose two datasets first"),(0,o.kt)("img",{src:"https://user-images.githubusercontent.com/59123869/155394693-8577f69b-59ea-43c7-9222-e71edc105fd3.png",width:"600"}),(0,o.kt)("h3",{id:"2-click-the-right-mouse-button-and-choose-the-compare"},"2. Click the right mouse button and choose the ",(0,o.kt)("inlineCode",{parentName:"h3"},"compare")),(0,o.kt)("img",{src:"https://user-images.githubusercontent.com/59123869/155394868-907abbd6-49dd-4455-85e6-c33c9a144c59.png",width:"200"}),(0,o.kt)("h3",{id:"3-datalab-will-generate-different-analytical-figures-and-tables-for-these-two-datasets"},"3. DataLab will generate different analytical figures and tables for these two datasets."),(0,o.kt)("img",{src:"https://user-images.githubusercontent.com/59123869/155395073-d1dfe0a2-8045-4b74-b78e-4a70729ea5f7.png",width:"600"}),(0,o.kt)("img",{src:"https://user-images.githubusercontent.com/59123869/155395692-6a610191-5588-4e82-881f-06df7adcb785.png",width:"600"}))}p.isMDXComponent=!0}}]);